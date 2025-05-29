# src/trainer/unlearn/nova.py
import torch
from torch import nn
import logging
from trainer.unlearn.base import UnlearnTrainer

# NOVA (Noise-Optimized Vector for Annulling)
# 
# Inspiration from 
# https://github.com/vikram2000b/Fast-Machine-Unlearning
# Extended in
# https://github.com/MoraiT01/study_on_unlearning # Here the algorithms, which follow these same principles, are refered to as GeFeU, FEMU and FEMU+

logger = logging.getLogger(__name__)

# Note:
# - I am assuming that the "inputs" (consiting of "input_ids", "attention_mask", "labels") are in the form of a batch and not a single sample.

class AntiPattern(nn.Module):
    def __init__(self, *dim, attention_mask):
        super().__init__()
        self.anti_pattern = torch.nn.Parameter(torch.randn(*dim), requires_grad = True)
        self.attention_mask = attention_mask
        
    def forward(self, attention_mask=None):
        if attention_mask is None:
            attention_mask = self.attention_mask
        masked_anti_pattern = self.anti_pattern * attention_mask
        return masked_anti_pattern

class NOVA(UnlearnTrainer):

    def __init__(self, 
            noise_epochs: int = 5,
            noise_lr: float = 0.2,
            regularization_term: float = 0.05,
            impair_gamma: float = 1.0,
            repair_alpha: float = 1.0,
            *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.noise_epochs = noise_epochs
        self.noise_lr = noise_lr
        self.regularization_term = regularization_term

        self.gamma = impair_gamma
        self.alpha = repair_alpha
    
    def train_anti_pattern(self, model, anti_pattern: AntiPattern, forget_surrogate_label):
        

        # Set the model to evaluation mode
        model.eval()
        # Initialize the optimizer and the learning rate scheduler
        optimizers = torch.optim.adam.Adam(anti_pattern.parameters(), lr = self.noise_lr)

        # Train the model
        logger.info("Training the anti-pattern...")
        for epoch in range(self.noise_epochs):
            input_batch = {
                "input_ids": anti_pattern(),
                "attention_mask": anti_pattern.attention_mask,
                "labels": forget_surrogate_label
            }
            
            # Forward pass
            outputs = model(**input_batch)
            # Calculate the loss
            loss = - outputs.loss + self.regularization_term * torch.mean(torch.sum(torch.square(input_batch["input_ids"])))
            # Backward pass
            optimizers.zero_grad()
            loss.backward()
            optimizers.step()

            logger.info(f"Epoch: {epoch+1}, Loss: {loss.item():.4f}")

        logger.info("Anti-pattern training complete.")
        return anti_pattern.anti_pattern.detach()


    def prepare_anti_pattern(self, model, forget_inputs, use_surrogate_labels=True):

        noise = AntiPattern(*forget_inputs["input_ids"].shape, attention_mask=forget_inputs["attention_mask"])

        if use_surrogate_labels:
            labels = model(**forget_inputs)
        else:
            labels = forget_inputs["labels"]

        trained_anti_patterns = self.train_anti_pattern(model, noise, labels)

        anti_pattern = {
            "input_ids": trained_anti_patterns,
            "attention_mask": forget_inputs["attention_mask"],
            "labels": labels,
        }
        return anti_pattern

    def compute_intermediate_loss(self, model, retain_inputs):
        retain_outputs = model(**retain_inputs)
        retain_loss = 0.0
        
        # Using NLL as an example
        retain_loss += retain_outputs.loss
        
        return retain_loss

    def compute_loss(self, model, inputs, return_outputs=False):
        forget_inputs = inputs["forget"]
        forget_inputs = {
            "input_ids": forget_inputs["input_ids"],
            "attention_mask": forget_inputs["attention_mask"],
            "labels": forget_inputs["labels"],
        }

        anti_pattern = self.prepare_anti_pattern(model, forget_inputs)

        anti_outputs = model(**anti_pattern)
        anti_loss = anti_outputs.loss

        retain_inputs = inputs["retain"]
        retain_inputs = {
            "input_ids": retain_inputs["input_ids"],
            "attention_mask": retain_inputs["attention_mask"],
            "labels": retain_inputs["labels"],
        }
        retain_loss = self.compute_intermediate_loss(model=model, retain_inputs=retain_inputs)

        loss = self.gamma * anti_loss + self.alpha * retain_loss

        return (loss, anti_outputs) if return_outputs else loss