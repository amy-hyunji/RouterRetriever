import os
import torch
import numpy as np
import pytorch_lightning as pl
import torch.distributed as dist
from transformers import AutoTokenizer, Adafactor, AdamW
from torch.utils.data import DataLoader, SequentialSampler

from data import ContrieverDataset
from src import contriever, utils, dist_utils
from peft import get_peft_config, get_peft_model, LoraConfig, TaskType, PeftModel

class FineTune_Contriever(pl.LightningModule):
    def __init__(self, hparams):
        super(FineTune_Contriever, self).__init__()

        self.save_hyperparameters(hparams)

        peft_config = LoraConfig(
            r=self.hparams.rank, lora_alpha=self.hparams.lora_alpha, lora_dropout=self.hparams.lora_dropout
        )

        q_model = contriever.Contriever.from_pretrained(hparams.model_name_or_path)
        ctx_model = contriever.Contriever.from_pretrained(hparams.model_name_or_path)
    
        self.q_model = get_peft_model(q_model, peft_config)
        
        if self.hparams.freeze_ctx:
            self.ctx_model =  self.freeze_model(ctx_model)
        else:
            self.ctx_model = get_peft_model(ctx_model, peft_config)

        if hparams.q_lora_ckpt is not None:
            print(f"Loading query lora ckpt from .. {hparams.q_lora_ckpt}")
            self.q_model = PeftModel.from_pretrained(self.q_model, hparams.q_lora_ckpt, adapter_name=self.hparams.peft_name)
        if hparams.ctx_lora_ckpt is not None:
            assert not self.hparams.freeze_ctx
            print(f"Loading context lora ckpt from .. {hparams.ctx_lora_ckpt}")
            self.ctx_model = PeftModel.from_pretrained(self.ctx_model, hparams.ctx_lora_ckpt, adapter_name=self.hparams.peft_name)
        
        self.q_model.print_trainable_parameters()
        if not self.hparams.freeze_ctx:
            self.ctx_model.print_trainable_parameters()
            
        self.tokenizer = AutoTokenizer.from_pretrained(hparams.tokenizer_name_or_path)

        if self.tokenizer.bos_token_id is None:
            self.tokenizer.bos_token = "[CLS]" 
        if self.tokenizer.eos_token_id is None:
            self.tokenizer.eos_token = "[SEP]"

        self.q_model.config.pooling = "average" 
        self.ctx_model.config.pooling = "average" 
        self.run_stats = utils.WeightedAvgStats()

        self.validation_step_outputs = []

    def freeze_model(self, model):
        for param in model.parameters():
            param.requires_grad = False
        return model

    def do_print(self, text):
        if torch.cuda.current_device() == 0:
            print(text)

    def train_dataloader(self):
        train_dataset = ContrieverDataset(
            datapaths=self.hparams.train_file,
            training=True,
            tokenizer=self.tokenizer,
            maxlength=self.hparams.max_length,
            negative_ctxs=self.hparams.negative_ctxs,
            negative_hard_ratio=self.hparams.negative_hard_ratio,
            negative_hard_min_idx=self.hparams.negative_hard_min_idx,
            normalize=self.hparams.eval_normalize_text,
        )
        self.do_print(f"# of training dataset: {len(train_dataset)}")
        train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=self.hparams.train_batch_size, drop_last=True, num_workers=self.hparams.num_workers)
        return train_dataloader

    def val_dataloader(self):
        eval_dataset = ContrieverDataset(
            datapaths=self.hparams.dev_file,
            training=False,
            tokenizer=self.tokenizer,
            maxlength=self.hparams.max_length,
            negative_ctxs=self.hparams.negative_ctxs,
            negative_hard_ratio=self.hparams.negative_hard_ratio,
            negative_hard_min_idx=self.hparams.negative_hard_min_idx,

            normalize=self.hparams.eval_normalize_text,
        )
        self.do_print(f"# of eval dataset: {len(eval_dataset)}")
        eval_dataloader = DataLoader(eval_dataset, shuffle=False, batch_size=self.hparams.train_batch_size, drop_last=False, num_workers=self.hparams.num_workers)
        return eval_dataloader

    def test_dataloader(self):
        eval_dataset = ContrieverDataset(
            datapaths=self.hparams.test_file,
            training=False,
            normalize=self.hparams.eval_normalize_text,
        )
        collator = Collator(self.tokenizer, maxlength=self.hparams.max_length)
        eval_sampler = SequentialSampler(eval_dataset)
        eval_dataloader = DataLoader(
            eval_dataset,
            sampler=eval_sampler,
            batch_size=self.hparams.train_batch_size,
            drop_last=False,
            num_workers=self.hparams.num_workers,
            collate_fn=collator,
        )
        return eval_dataloader

    def forward(self, q_tokens, q_mask, g_tokens, g_mask, n_tokens, n_mask, stats_prefix="", iter_stats={}, **kwargs):

        if self.hparams.negative_ctxs == 0:
            k_mask = g_mask 
            k_tokens = g_tokens
        else:
            k_mask = torch.cat([g_mask, n_mask], dim=0)
            k_tokens = torch.cat([g_tokens, n_tokens], dim=0) 
         
        bsz = len(q_tokens)
        labels = torch.arange(0, bsz, dtype=torch.long, device=q_tokens.device)

        qemb = self.q_model(input_ids=q_tokens, attention_mask=q_mask, normalize=self.hparams.eval_normalize_text)
        kemb = self.ctx_model(input_ids=k_tokens, attention_mask=k_mask, normalize=self.hparams.eval_normalize_text)

        gather_fn = dist_utils.gather
        gather_kemb = gather_fn(kemb)
        labels = labels + dist_utils.get_rank() * len(kemb)

        scores = torch.einsum("id, jd->ij", qemb / self.hparams.temperature, gather_kemb)
        loss = torch.nn.functional.cross_entropy(scores, labels, label_smoothing=self.hparams.label_smoothing)

        # log stats
        if len(stats_prefix) > 0:
            stats_prefix = stats_prefix + "/"
        iter_stats[f"{stats_prefix}loss"] = (loss.item(), bsz)

        predicted_idx = torch.argmax(scores, dim=-1)
        accuracy = 100 * (predicted_idx == labels).float().mean()
        stdq = torch.std(qemb, dim=0).mean().item()
        stdk = torch.std(kemb, dim=0).mean().item()
        iter_stats[f"{stats_prefix}accuracy"] = (accuracy, bsz)
        iter_stats[f"{stats_prefix}stdq"] = (stdq, bsz)
        iter_stats[f"{stats_prefix}stdk"] = (stdk, bsz)

        self.log(f"{stats_prefix}accuracy", accuracy, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log(f"{stats_prefix}stdq", stdq, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log(f"{stats_prefix}stdk", stdk, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

        # print(f"Loss!!! {loss}")

        return loss, iter_stats

    def training_step(self, batch, batch_idx):

        self.q_model.train()
        if not self.hparams.freeze_ctx:
            self.ctx_model.train()
        #self.optimizer.zero_grad()

        if type(batch["n_tokens"]) != list and len(batch["n_tokens"].shape) == 3:
            dim = list(batch["n_tokens"].shape)[-1]
            batch["n_tokens"] = batch["n_tokens"].view(-1, dim)
            batch["n_mask"] = batch["n_mask"].view(-1, dim)

        train_loss, iter_stats = self(**batch, stats_prefix="train")
        self.run_stats.update(iter_stats)

        #self.manual_backward(train_loss)
        #self.optimizer.step()
        #self.scheduler.step()

        return train_loss

    def validation_step(self, batch, batch_idx):
        
        self.q_model.eval()
        self.ctx_model.eval()
        # batch = {key: value.cuda() if isinstance(value, torch.Tensor) else value for key, value in batch.items()}
        if self.hparams.dev_negative_ctxs == 0:
            all_tokens = batch["g_tokens"]
            all_mask = batch["g_mask"]
        else:
            all_tokens = torch.cat([batch["g_tokens"], batch["n_tokens"]], dim=0)
            all_mask = torch.cat([batch["g_mask"], batch["n_mask"]], dim=0)

        q_emb = self.q_model(input_ids=batch["q_tokens"], attention_mask=batch["q_mask"], normalize=self.hparams.eval_normalize_text)
        all_emb = self.ctx_model(input_ids=all_tokens, attention_mask=all_mask, normalize=self.hparams.eval_normalize_text)
        g_emb, n_emb = torch.split(all_emb, [len(batch["g_tokens"]), len(batch["n_tokens"])])

        ret = {"q_emb": q_emb, "g_emb": g_emb, "n_emb": n_emb}
        self.validation_step_outputs.append(ret)
        return

    def on_validation_epoch_end(self):

        all_q = torch.cat([x["q_emb"] for x in self.validation_step_outputs], dim=0)
        all_g = torch.cat([x["g_emb"] for x in self.validation_step_outputs], dim=0)
        all_n = torch.cat([x["n_emb"] for x in self.validation_step_outputs], dim=0)

        labels = torch.arange(0, len(all_q), device=all_q.device, dtype=torch.long)

        all_sizes = dist_utils.get_varsize(all_g)
        all_g = dist_utils.varsize_gather_nograd(all_g)
        all_n = dist_utils.varsize_gather_nograd(all_n)
        labels = labels + sum(all_sizes[: dist_utils.get_rank()])

        scores_pos = torch.einsum("id, jd->ij", all_q, all_g)
        scores_neg = torch.einsum("id, jd->ij", all_q, all_n)
        scores = torch.cat([scores_pos, scores_neg], dim=-1)

        argmax_idx = torch.argmax(scores, dim=1)
        sorted_scores, indices = torch.sort(scores, descending=True)
        isrelevant = indices == labels[:, None]
        rs = [r.cpu().numpy().nonzero()[0] for r in isrelevant]
        mrr = np.mean([1.0 / (r[0] + 1) if r.size else 0.0 for r in rs])

        acc = (argmax_idx == labels).sum() / all_q.size(0)
        acc, total = dist_utils.weighted_average(acc, all_q.size(0))
        mrr, _ = dist_utils.weighted_average(mrr, all_q.size(0))
        acc = 100 * acc

        self.log('val_eval_acc', acc, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=False)
        self.log('val_eval_mrr', mrr, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=False)

        if dist_utils.is_main():
            message = [f"eval acc: {acc:.2f}%", f"eval mrr: {mrr:.3f}"]
            print(" | ".join(message))
            # if tb_logger is not None:
            #     tb_logger.add_scalar(f"eval_acc", acc, step)
            #     tb_logger.add_scalar(f"mrr", mrr, step)
        self.validation_step_outputs = []
        save_path = os.path.join(self.hparams.output_dir, f"best_tfmr_{self.current_epoch}")
        self._save_checkpoint()

    def test_step(self, batch, batch_idx):
        batch = {key: value.cuda() if isinstance(value, torch.Tensor) else value for key, value in batch.items()}

        all_tokens = torch.cat([batch["g_tokens"], batch["n_tokens"]], dim=0)
        all_mask = torch.cat([batch["g_mask"], batch["n_mask"]], dim=0)

        q_emb = self.q_model(input_ids=batch["q_tokens"], attention_mask=batch["q_mask"], normalize=self.hparams.eval_normalize_text)
        all_emb = self.ctx_model(input_ids=all_tokens, attention_mask=all_mask, normalize=self.hparams.eval_normalize_text)

        g_emb, n_emb = torch.split(all_emb, [len(batch["g_tokens"]), len(batch["n_tokens"])])

        ret = {"q_emb": q_emb, "g_emb": g_emb, "n_emb": n_emb}
        self.validation_step_outputs.append(ret)
        return

    def on_test_epoch_end(self):

        all_q = torch.cat([x["q_emb"] for x in self.validation_step_outputs], dim=0)
        all_g = torch.cat([x["g_emb"] for x in self.validation_step_outputs], dim=0)
        all_n = torch.cat([x["n_emb"] for x in self.validation_step_outputs], dim=0)

        labels = torch.arange(0, len(all_q), device=all_q.device, dtype=torch.long)

        all_sizes = dist_utils.get_varsize(all_g)
        all_g = dist_utils.varsize_gather_nograd(all_g)
        all_n = dist_utils.varsize_gather_nograd(all_n)
        labels = labels + sum(all_sizes[: dist_utils.get_rank()])

        scores_pos = torch.einsum("id, jd->ij", all_q, all_g)
        scores_neg = torch.einsum("id, jd->ij", all_q, all_n)
        scores = torch.cat([scores_pos, scores_neg], dim=-1)

        argmax_idx = torch.argmax(scores, dim=1)
        sorted_scores, indices = torch.sort(scores, descending=True)
        isrelevant = indices == labels[:, None]
        rs = [r.cpu().numpy().nonzero()[0] for r in isrelevant]
        mrr = np.mean([1.0 / (r[0] + 1) if r.size else 0.0 for r in rs])

        acc = (argmax_idx == labels).sum() / all_q.size(0)
        acc, total = dist_utils.weighted_average(acc, all_q.size(0))
        mrr, _ = dist_utils.weighted_average(mrr, all_q.size(0))
        acc = 100 * acc

        self.log('val_eval_acc', acc, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log('val_eval_mrr', mrr, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

        if dist_utils.is_main():
            message = [f"eval acc: {acc:.2f}%", f"eval mrr: {mrr:.3f}"]
            print(" | ".join(message))
            # if tb_logger is not None:
            #     tb_logger.add_scalar(f"eval_acc", acc, step)
            #     tb_logger.add_scalar(f"mrr", mrr, step)
        self.validation_step_outputs.clear()

    def _set_optimizer(self):
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.q_model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": [p for n, p in self.q_model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]

        optimizer = Adafactor(optimizer_grouped_parameters, lr=self.hparams.learning_rate, warmup_init=False, scale_parameter=False, relative_step=False,)
        print(f"**** Set optimizer!!!")
        return optimizer

    def configure_optimizers(self):
        
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.q_model.named_parameters() if not any(nd in n for nd in no_decay)], 
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": [p for n, p in self.q_model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        
        optimizer = torch.optim.AdamW(
            self.q_model.parameters(), lr=self.hparams.learning_rate, betas=(0.9, 0.999) 
        )
        self.opt = optimizer 
        print(f"**** Set optimizer!!!")
       
        if self.hparams.lr_scheduler == 'linear':
            return [optimizer]
        elif self.hparams.lr_scheduler == "exponential":
            len_data = len(self.train_dataloader())
            denominator=self.hparams.n_gpu
            steps_per_epoch=((len_data//denominator)+1)//self.hparams.gradient_accumulation_steps
            scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=self.hparams.learning_rate, steps_per_epoch=steps_per_epoch, pct_start=0.1, epochs=self.hparams.num_train_epochs, anneal_strategy= 'linear', cycle_momentum=False)
            return [optimizer], [{'scheduler': scheduler, 'interval': 'step', 'name': 'learning_rate'}] 
        else:
            assert False

    def _save_checkpoint(self):
        save_path = os.path.join(self.hparams.output_dir, f"best_tfmr_{self.current_epoch}")
        print(f"Save.. {self.current_epoch} in {save_path}")
        os.makedirs(self.hparams.output_dir, exist_ok=True)
        self.q_model.save_pretrained(os.path.join(save_path, "q"))
        self.tokenizer.save_pretrained(save_path)
        if not self.hparams.freeze_ctx:
            self.ctx_model.save_pretrained(os.path.join(save_path, "ctx"))
