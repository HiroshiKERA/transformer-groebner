import torch
import wandb
from typing import Optional
from transformers import Trainer, TrainingArguments

class CustomTrainingArguments(TrainingArguments):
    def __init__(self, *args, max_steps_per_epoch: Optional[int] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_steps_per_epoch = max_steps_per_epoch

class CustomTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.log_history = []

    def compute_loss(self, model, inputs, return_outputs=False):
        outputs = model(**inputs)
        loss = outputs['loss']

        # Log to WandB
        # if self.args.logging_steps > 0 and self.state.global_step % self.args.logging_steps == 0:
        ignore_index = model.special_token_ids['pad_token_id']
        self.log_metrics(outputs, inputs, ignore_index=ignore_index)

        return (loss, outputs) if return_outputs else loss

    def log_metrics(self, outputs, inputs, ignore_index=-100):
        if not self.is_world_process_zero():
                return
        
        metrics = {
            "train/loss": outputs['loss'].mean().item(),
            "train/loss_clf": outputs['loss_clf'].mean().item(),
            "train/loss_rg": outputs['loss_rg'].mean().item()
        }

        # Calculate accuracy (if needed)
        if 'labels' in inputs and 'logits' in outputs:
            labels = inputs['labels']
            labels, logits = labels[labels != ignore_index], outputs['logits'][labels != ignore_index]
            predictions = torch.argmax(logits, dim=-1)
            metrics["train/error"] = (predictions != labels).float().mean().item()

        # Add to log history
        self.log_history.append(metrics)

        # Log to WandB
        # wandb.log(metrics, step=self.state.global_step)
        wandb.log(metrics)