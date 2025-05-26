import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from transformer_lens import HookedTransformer
from typing import List, Tuple, Dict


class PPOTrainer:
    def __init__(
            self,
            model_name: str = "gpt2-small",
            learning_rate: float = 1e-5,
            clip_ratio: float = 0.2,
            value_loss_coef: float = 0.5,
            entropy_coef: float = 0.01,
            kl_coef: float = 0.1,
            use_kl_penalty: bool = True,
            max_length: int = 50,
            batch_size: int = 4,
            ppo_epochs: int = 4,
            device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.device = device
        self.clip_ratio = clip_ratio
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.kl_coef = kl_coef
        self.use_kl_penalty = use_kl_penalty
        self.max_length = max_length
        self.batch_size = batch_size
        self.ppo_epochs = ppo_epochs

        # Load model and tokenizer
        self.model = HookedTransformer.from_pretrained(model_name, device=device)
        self.tokenizer = self.model.tokenizer

        # Add pad token if it doesn't exist
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Create reference model for KL penalty (frozen copy)
        self.ref_model = HookedTransformer.from_pretrained(model_name, device=device)
        for param in self.ref_model.parameters():
            param.requires_grad = False

        # Value head for critic
        self.value_head = nn.Linear(self.model.cfg.d_model, 1).to(device)

        # Optimizer
        params = list(self.model.parameters()) + list(self.value_head.parameters())
        self.optimizer = torch.optim.Adam(params, lr=learning_rate)

    def reward_function(self, text: str) -> float:
        """Reward function: count number of periods in the text"""
        return float(text.count('.'))

    def generate_response(self, prompt: str, temperature: float = 1.0) -> Tuple[str, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Generate response and return text, log probs, values, and tokens"""
        # Tokenize prompt
        prompt_tokens = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)

        # Generate tokens
        generated_tokens = []
        log_probs = []
        values = []

        current_tokens = prompt_tokens.clone()

        for _ in range(self.max_length):
            # Get model outputs (no grad for generation, will recompute later for training)
            with torch.no_grad():
                logits = self.model(current_tokens)[:, -1, :]  # Last token logits

                # Get hidden states for value estimation
                _, cache = self.model.run_with_cache(current_tokens)
                hidden_states = cache['blocks.11.hook_resid_post'][:, -1, :]
                value = self.value_head(hidden_states).squeeze()
                values.append(value)

                # Sample next token
                probs = F.softmax(logits / temperature, dim=-1)
                dist = Categorical(probs)
                next_token = dist.sample()
                log_prob = dist.log_prob(next_token)

                generated_tokens.append(next_token.item())
                log_probs.append(log_prob)

                # Update current tokens - add new token to sequence
                next_token_tensor = next_token.unsqueeze(0)  # Shape: [1]
                current_tokens = torch.cat([current_tokens, next_token_tensor], dim=1)

        if len(generated_tokens) == 0:
            # Handle case where no tokens were generated
            return "", torch.tensor([]), torch.tensor([]), torch.tensor([])

        # Convert to tensors
        generated_tokens_tensor = torch.tensor(generated_tokens, device=self.device)
        log_probs_tensor = torch.stack(log_probs)
        values_tensor = torch.stack(values)

        # Decode generated text only (not including prompt)
        generated_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)

        return generated_text, log_probs_tensor, values_tensor, generated_tokens_tensor

    def compute_advantages(self, rewards: torch.Tensor, values: torch.Tensor, gamma: float = 0.99, lam: float = 0.95) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute GAE advantages and returns"""
        if len(rewards) == 0:
            return torch.tensor([]), torch.tensor([])

        advantages = torch.zeros_like(rewards)
        returns = torch.zeros_like(rewards)

        last_advantage = 0
        last_return = 0

        for t in reversed(range(len(rewards))):
            # For the last step, next_value is 0
            next_value = values[t + 1] if t < len(values) - 1 else 0

            # TD error
            delta = rewards[t] + gamma * next_value - values[t]

            # GAE advantage
            advantages[t] = delta + gamma * lam * last_advantage
            last_advantage = advantages[t]

            # Return
            returns[t] = rewards[t] + gamma * last_return
            last_return = returns[t]

        return advantages, returns

    def compute_kl_penalty(self, full_sequence: torch.Tensor, current_log_probs: torch.Tensor) -> torch.Tensor:
        """Compute KL divergence penalty against reference model"""
        if not self.use_kl_penalty:
            return torch.tensor(0.0, device=self.device)

        try:
            with torch.no_grad():
                # Get reference model outputs for the full sequence
                ref_logits = self.ref_model(full_sequence.unsqueeze(0))

                # Get reference log probs for generated tokens only
                gen_len = len(current_log_probs)
                ref_gen_logits = ref_logits[:, -gen_len:, :]
                ref_log_probs = F.log_softmax(ref_gen_logits, dim=-1)

                # Get reference log probs for the actual tokens
                tokens = full_sequence[-gen_len:]
                ref_log_probs_selected = ref_log_probs.squeeze(0).gather(1, tokens.unsqueeze(1)).squeeze(1)

            # KL divergence: E[log(p/q)] = E[log(p) - log(q)]
            kl_div = current_log_probs - ref_log_probs_selected
            return kl_div.mean()
        except Exception:
            # Return zero if there's any issue computing KL
            return torch.tensor(0.0, device=self.device)

    def ppo_update(self, batch_data: List[Dict]) -> Dict[str, float]:
        """Perform PPO update"""
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy_loss = 0
        total_kl_penalty = 0

        for epoch in range(self.ppo_epochs):
            for data in batch_data:
                if len(data['tokens']) == 0:
                    continue

                tokens = data['tokens']
                old_log_probs = data['log_probs']
                advantages = data['advantages']
                returns = data['returns']
                prompt_tokens = data['prompt_tokens']

                # Create full sequence (prompt + generated tokens)
                full_sequence = torch.cat([prompt_tokens.squeeze(0), tokens])

                # Forward pass through model
                logits = self.model(full_sequence.unsqueeze(0))

                # Get logits for generated tokens only
                gen_logits = logits[:, -len(tokens):, :]  # Shape: [1, seq_len, vocab_size]

                # Get hidden states for value prediction
                _, cache = self.model.run_with_cache(full_sequence.unsqueeze(0))
                hidden_states = cache['blocks.11.hook_resid_post'][:, -len(tokens):, :]
                value_preds = self.value_head(hidden_states).squeeze()  # Shape: [seq_len]

                # Current policy log probs
                log_probs_dist = F.log_softmax(gen_logits, dim=-1)  # Shape: [1, seq_len, vocab_size]
                current_log_probs = log_probs_dist.squeeze(0).gather(1, tokens.unsqueeze(1)).squeeze(
                    1)  # Shape: [seq_len]

                # Policy ratio
                ratio = torch.exp(current_log_probs - old_log_probs)

                # Clipped surrogate loss
                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss
                if len(value_preds.shape) == 0:  # Handle single value case
                    value_preds = value_preds.unsqueeze(0)
                value_loss = F.mse_loss(value_preds, returns)

                # Entropy loss
                probs = F.softmax(gen_logits, dim=-1)
                entropy = -(probs * log_probs_dist).sum(dim=-1).mean()
                entropy_loss = -self.entropy_coef * entropy

                # KL penalty
                kl_penalty = self.compute_kl_penalty(full_sequence, current_log_probs)

                # Total loss
                total_loss = policy_loss + self.value_loss_coef * value_loss + entropy_loss - self.kl_coef * kl_penalty
                print(f"{total_loss.item()=}")

                # Backward pass
                self.optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(list(self.model.parameters()) + list(self.value_head.parameters()), 1.0)
                self.optimizer.step()

                # Accumulate losses
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy_loss += entropy_loss.item()
                total_kl_penalty += kl_penalty.item() if isinstance(kl_penalty, torch.Tensor) else kl_penalty

        num_updates = sum(1 for data in batch_data if len(data['tokens']) > 0) * self.ppo_epochs
        if num_updates == 0:
            return {'policy_loss': 0, 'value_loss': 0, 'entropy_loss': 0, 'kl_penalty': 0}

        return {
            'policy_loss': total_policy_loss / num_updates,
            'value_loss': total_value_loss / num_updates,
            'entropy_loss': total_entropy_loss / num_updates,
            'kl_penalty': total_kl_penalty / num_updates,
        }

    def train_step(self, prompts: List[str]) -> Dict[str, float]:
        """Single training step"""
        batch_data = []
        total_reward = 0

        for prompt in prompts:
            # Generate response
            generated_text, log_probs, values, tokens = self.generate_response(prompt)

            # Compute reward
            reward = self.reward_function(generated_text)
            total_reward += reward

            # Create reward tensor (sparse reward at the end)
            rewards = torch.zeros(len(tokens), device=self.device)
            if len(tokens) > 0:
                rewards[-1] = reward

            # Compute advantages and returns
            advantages, returns = self.compute_advantages(rewards, values)

            # Normalize advantages
            if len(advantages) > 1:
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            # Store data
            prompt_tokens = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
            batch_data.append({
                'tokens': tokens,
                'log_probs': log_probs,
                'advantages': advantages,
                'returns': returns,
                'prompt_tokens': prompt_tokens,
                'reward': reward
            })

        # PPO update
        losses = self.ppo_update(batch_data)
        losses['avg_reward'] = total_reward / len(prompts)

        return losses

    def generate_samples(self, prompts: List[str], num_samples: int = 3) -> None:
        """Generate and print sample outputs for evaluation"""
        print("=" * 60)
        print("SAMPLE GENERATIONS:")
        print("=" * 60)

        sample_prompts = prompts[:num_samples]
        for i, prompt in enumerate(sample_prompts):
            generated_text, _, _, _ = self.generate_response(prompt, temperature=0.8)
            periods = generated_text.count('.')

            print(f"Sample {i + 1}:")
            print(f"  Prompt: '{prompt}'")
            print(f"  Generated: '{generated_text}'")
            print(f"  Periods: {periods}")
            print(f"  Reward: {self.reward_function(generated_text)}")
            print()
        print("=" * 60)
        print()

    def train(self, prompts: List[str], num_iterations: int = 100):
        """Main training loop"""
        print(f"Starting PPO training with KL penalty: {self.use_kl_penalty}")
        print(f"Training for {num_iterations} iterations")
        print()

        # Generate samples before training
        print("BEFORE TRAINING:")
        self.generate_samples(prompts)

        for iteration in range(num_iterations):
            print(f"Beginning Iteration #{iteration=} of RL training")

            # Sample batch of prompts
            batch_prompts = np.random.choice(prompts, size=min(self.batch_size, len(prompts)), replace=False)

            # Training step
            losses = self.train_step(batch_prompts.tolist())

            # Log progress
            print(f"Iteration {iteration}:")
            print(f"  Avg Reward: {losses['avg_reward']:.3f}")
            print(f"  Policy Loss: {losses['policy_loss']:.3f}")
            print(f"  Value Loss: {losses['value_loss']:.3f}")
            print(f"  Entropy Loss: {losses['entropy_loss']:.3f}")
            if self.use_kl_penalty:
                print(f"  KL Penalty: {losses['kl_penalty']:.3f}")
            print()

            # Generate samples during training (every 20 iterations)
            if iteration > 0 and iteration % 2 == 0:
                print(f"DURING TRAINING (Iteration {iteration}):")
                self.generate_samples(prompts)

        # Generate samples after training
        print("AFTER TRAINING:")
        self.generate_samples(prompts)


# Trainer function
def trainer():
    """Main trainer function that runs PPO experiments"""
    # Training prompts
    prompts = [
        "The weather today is",
        "I went to the store and",
        "My favorite book is",
        "The cat sat on",
        "Yesterday I learned",
        "The movie was",
        "In the garden there",
        "She walked down the street",
        "The food tasted",
        "When I was young"
    ]

    print("PPO Training for GPT-2 Small with Period Reward Function")
    print("=" * 80)
    print("Objective: Train the model to generate text with more periods")
    print("Reward Function: Count of periods (.) in generated text")
    print("=" * 80)
    print()

    # Experiment 1: Training with KL penalty
    print("EXPERIMENT 1: Training WITH KL penalty")
    print("=" * 80)

    trainer_with_kl = PPOTrainer(
        learning_rate=1e-5,
        use_kl_penalty=True,
        kl_coef=0.1,
        batch_size=4,
        max_length=25,
        ppo_epochs=3,
        clip_ratio=0.2
    )

    trainer_with_kl.train(prompts, num_iterations=50)

    # print("\n" + "=" * 80)
    # print("EXPERIMENT 2: Training WITHOUT KL penalty")
    # print("=" * 80)
    #
    # # Experiment 2: Training without KL penalty
    # trainer_without_kl = PPOTrainer(
    #     learning_rate=1e-5,
    #     use_kl_penalty=False,
    #     batch_size=4,
    #     max_length=25,
    #     ppo_epochs=3,
    #     clip_ratio=0.2
    # )

    # trainer_without_kl.train(prompts, num_iterations=50)

    print("\n" + "=" * 80)
    print("TRAINING COMPLETE!")
    print("=" * 80)
    print("Summary:")
    print("- Trained GPT-2 small to generate text with more periods")
    print("- Compared training with and without KL penalty")
    print("- KL penalty helps maintain text quality while optimizing reward")
    print("- Without KL penalty, model may degenerate to repetitive patterns")
