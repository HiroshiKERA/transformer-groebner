import numpy as np
from sklearn.metrics import accuracy_score
from transformers import TrainerCallback, TrainerControl, TrainerState
from transformers import TrainingArguments

def preprocess_logits_for_metrics(outputs, labels):
    
    if len(outputs) == 3:
        logits, logits_for_regression = outputs[0], None
    if len(outputs) == 4:
        logits, logits_for_regression = outputs[0], outputs[1]  # for bart model
    
    # logits, logits_for_regression = outputs[2], outputs[3]  # for transformer_base
    
    predicted_ids = logits.argmax(dim=-1)
    
    return (predicted_ids, logits_for_regression)

def compute_metrics(eval_preds, ignore_index=-100):
    '''
    every thing is in numpy array format
    '''
    (predicted_ids, logits_for_regression), labels = eval_preds
    if isinstance(labels, tuple):
        labels, labels_for_regression = labels[0], labels[-1]
        
    predicted_ids = predicted_ids[labels != ignore_index]
    labels = labels[labels != ignore_index]

    error = 1 - accuracy_score(labels, predicted_ids)
    metrics = {"error rate": error}

    if logits_for_regression is not None and len(logits_for_regression) > 0:
        mse_fn = lambda x, y: np.mean((x - y)**2)
        is_continuous_tokens = np.isfinite(labels_for_regression)
        metrics[f"MSE"] = mse_fn(logits_for_regression[is_continuous_tokens], labels_for_regression[is_continuous_tokens])

    return metrics


class LimitStepsCallback(TrainerCallback):
    def __init__(self, max_steps_per_epoch: int):
        self.max_steps_per_epoch = max_steps_per_epoch if max_steps_per_epoch > 0 else float('inf')
        self.current_epoch_steps = 0

    def on_epoch_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        self.current_epoch_steps = 0

    def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        self.current_epoch_steps += 1
        if self.current_epoch_steps >= self.max_steps_per_epoch:
            control.should_epoch_stop = True
            control.should_save = True
            control.should_evaluate = True