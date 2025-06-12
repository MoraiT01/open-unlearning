# src/trainer/unlearn/nova.py
import torch
from torch import nn
import logging
from trainer.unlearn.base import UnlearnTrainer
import copy 

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
    def __init__(self, batch_size: int, seq_len: int, embedding_dim: int, attention_mask: torch.Tensor = None):
        super().__init__()
        # The core learned anti-pattern: initialized randomly with batch_size, seq_len, embedding_dim.
        # This parameter represents the direct perturbation for the input sequence, acting as a replacement for original embeddings.
        self.pattern = torch.nn.Parameter(torch.randn((batch_size, seq_len, embedding_dim)), requires_grad = True)
        self.mask = attention_mask.clone() if attention_mask is not None else None # Store mask as part of the instance

    def forward(self,):
        # Apply attention mask to the learned pattern
        if self.mask is not None:
            # Unsqueeze attention_mask to (batch_size, seq_len, 1) to enable broadcasting
            masked_pattern = self.pattern * self.mask.unsqueeze(-1)
            return masked_pattern
        return self.pattern

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

    def _optimize_anti_pattern_for_batch(self, model, forget_inputs):
        """
        Optimizes a *new* AntiPattern instance for the current batch of forget_inputs.
        This phase aims to maximize the model's loss on the forget set when perturbed by this batch-specific anti-pattern.
        The model is temporarily set to eval mode during this optimization.

        Returns:
            torch.Tensor: The optimized perturbation tensor for the current batch, detached from the graph.
        """
        # Store original model training state and set to eval mode
        original_training_state = model.training
        model.eval() # Ensure the main model's parameters are frozen during anti-pattern optimization

        # Get current batch dimensions and embedding dimension
        batch_size, seq_len = forget_inputs["input_ids"].shape
        embedding_dim = model.config.hidden_size # Get embedding dimension from model config

        # Create a NEW AntiPattern instance for this specific batch.
        # Its 'pattern' parameter will be initialized randomly for this batch.
        anti_pattern_instance = AntiPattern(
            batch_size=batch_size,
            seq_len=seq_len,
            embedding_dim=embedding_dim,
            attention_mask=forget_inputs["attention_mask"].to(model.device) # Ensure mask is on device
        ).to(model.device) # Move the AntiPattern instance to the correct device

        # Create a NEW optimizer for this specific AntiPattern instance.
        # This optimizer will only update parameters of 'anti_pattern_instance'.
        optimizer_for_this_batch = torch.optim.Adam(anti_pattern_instance.parameters(), lr=self.noise_lr)

        with torch.enable_grad(): # Ensure gradients are enabled for anti_pattern_instance parameters
            for _ in range(self.noise_epochs): # Loop for 'noise_epochs' optimization steps for this batch
                optimizer_for_this_batch.zero_grad()

                # --- CRITICAL CHANGE: The perturbation now REPLACES the original embeddings ---
                # 1. Generate perturbation (noise) from the batch-specific anti-pattern instance
                perturbation = anti_pattern_instance() # No arguments needed for forward() now

                # 2. Forward pass the model directly with the perturbation as input embeddings
                #    No addition to original_embeddings, as the perturbation acts as the full input.
                outputs = model(
                    inputs_embeds=perturbation, # Use perturbation directly as input embeddings
                    attention_mask=forget_inputs["attention_mask"],
                    labels=forget_inputs["labels"] # Use original labels to guide anti-pattern (maximize loss)
                )

                # Calculate anti-pattern loss: maximize loss on forget set + regularization
                # The `-` aims to maximize this loss (minimize negative loss).
                # .detach() is used for the regularization term as it penalizes the magnitude
                # of the perturbation directly, without backprop through the model's loss.
                anti_pattern_loss = - outputs.loss + self.regularization_term * torch.mean(torch.square(perturbation.detach()))

                # 3. Backpropagate and update AntiPattern parameters for this batch
                anti_pattern_loss.backward()
                optimizer_for_this_batch.step()
        
        # Restore model training state
        model.train(original_training_state)
        print(f"--- Anti-pattern Training is done ---")
        # Return the optimized perturbation tensor for the current batch
        return anti_pattern_instance.pattern.detach()

    def compute_intermediate_loss(self, model, inputs, uses_embeds: bool = False):
        
        if uses_embeds:
            outputs = model(
                inputs_embeds=inputs["input_ids"], 
                attention_mask=inputs["attention_mask"],
                labels=inputs["labels"] 
            )
    
        else:
            outputs = model(**inputs)
        loss = 0.0
        # Using NLL as an example
        loss += outputs.loss
        return loss

    def compute_loss(self, model, inputs, return_outputs=False):
        forget_inputs = inputs["forget"]
        retain_inputs = inputs["retain"]

        # --- Phase 1: Optimize a batch-specific AntiPattern and get its output ---
        # This function now returns the optimized perturbation for the current batch.
        # The AntiPattern instance and its optimizer are created and used within _optimize_anti_pattern_for_batch.
        optimized_perturbation_for_batch = self._optimize_anti_pattern_for_batch(model, forget_inputs)


        # --- Phase 2: Main Model Unlearning Step using the optimized perturbation from Phase 1 ---
        # This forward pass also uses perturbed embeddings, applying the effect of the anti-pattern.

        # The optimized_perturbation_for_batch now directly acts as the noisy input embeddings.
        final_antipattern_embeddings = optimized_perturbation_for_batch 

        # Compute forget loss (using the antipatterns as the input)
        forget_inputs_processed ={
            "input_ids": final_antipattern_embeddings,
            "attention_mask": forget_inputs["attention_mask"],
            "labels": forget_inputs["labels"],
        }
        forget_loss = self.compute_intermediate_loss(model=model, inputs=forget_inputs_processed, uses_embeds=True)

        # Compute retain loss (using standard input_ids)
        retain_inputs_processed = {
            "input_ids": retain_inputs["input_ids"],
            "attention_mask": retain_inputs["attention_mask"],
            "labels": retain_inputs["labels"],
        }
        retain_loss = self.compute_intermediate_loss(model=model, inputs=retain_inputs_processed)

        # Combine losses for the main model's update
        loss = self.gamma * forget_loss + self.alpha * retain_loss

        return (loss, forget_outputs) if return_outputs else loss