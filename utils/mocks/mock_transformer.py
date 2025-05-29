import torch
from dataclasses import dataclass
from typing import Optional, Union, Tuple, List


@dataclass
class MockModelOutput:
    """Mock output class for generate method"""
    sequences: torch.Tensor
    scores: Optional[Tuple[torch.Tensor]] = None
    attentions: Optional[Tuple[torch.Tensor]] = None
    hidden_states: Optional[Tuple[torch.Tensor]] = None


class MockConfig:
    """Mock configuration class, takes Qwen 2.5 parameter defaults"""

    def __init__(
            self,
            name_or_path: str = "mock-model",
            vocab_size: int = 151936,
            hidden_size: int = 4096,
            intermediate_size: int = 22016,
            num_hidden_layers: int = 32,
            num_attention_heads: int = 32,
            num_key_value_heads: int = 32,
            hidden_act: str = 'silu',
            max_position_embeddings: int = 32768,
            initializer_range: float = 0.02,
            rms_norm_eps: float = 1e-06,
            use_cache: bool = True,
            tie_word_embeddings: bool = False,
            rope_theta: float = 10000.0,
            rope_scaling=None,
            use_sliding_window: bool = False,
            sliding_window: int = 4096,
            max_window_layers: int = 28,
            layer_types=None,
            attention_dropout: float = 0.0,
            **kwargs
    ):
        self.name_or_path = name_or_path
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.tie_word_embeddings = tie_word_embeddings
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self.use_sliding_window = use_sliding_window
        self.sliding_window = sliding_window
        self.max_window_layers = max_window_layers
        self.layer_types = layer_types
        self.attention_dropout = attention_dropout
        # Store any additional config parameters
        for key, value in kwargs.items():
            setattr(self, key, value)


class MockAutoModelForCausalLM:
    """Mock class that mimics HuggingFace AutoModelForCausalLM"""

    def __init__(self, config: Optional[MockConfig] = None):
        self.config = config or MockConfig()
        self.device = torch.device("cpu")

    def to(self, device: Union[str, torch.device]):
        """Move model to specified device"""
        if isinstance(device, str):
            device = torch.device(device)
        self.device = device
        return self

    def eval(self):
        """Set model to evaluation mode"""
        return self

    def train(self, mode: bool = True):
        """Set model to training mode"""
        return self

    def generate(
            self,
            input_ids: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            max_new_tokens: Optional[int] = 20,
            max_length: Optional[int] = None,
            min_length: Optional[int] = 0,
            do_sample: bool = True,
            temperature: float = 1.0,
            top_k: Optional[int] = 50,
            top_p: Optional[float] = 1.0,
            pad_token_id: Optional[int] = None,
            eos_token_id: Optional[int] = None,
            return_dict_in_generate: bool = False,
            output_scores: bool = False,
            output_attentions: bool = False,
            output_hidden_states: bool = False,
            **kwargs
    ) -> Union[torch.Tensor, MockModelOutput]:
        """
        Mock generate method that produces random tokens
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device

        # Calculate total length
        if max_length is None:
            max_length = seq_len + (max_new_tokens or 20)

        # Generate random tokens for new positions
        new_tokens_count = min(max_new_tokens or (max_length - seq_len), max_length - seq_len)

        if new_tokens_count > 0:
            # Generate random token ids
            new_tokens = torch.randint(
                low=0,
                high=self.config.vocab_size,
                size=(batch_size, new_tokens_count),
                device=device
            )

            # Concatenate with input
            generated_sequences = torch.cat([input_ids, new_tokens], dim=1)
        else:
            generated_sequences = input_ids

        # Generate mock scores if requested
        scores = None
        if output_scores:
            scores = tuple(
                torch.randn(batch_size, self.config.vocab_size, device=device)
                for _ in range(new_tokens_count)
            )

        # Return based on return_dict_in_generate flag
        if return_dict_in_generate:
            return MockModelOutput(
                sequences=generated_sequences,
                scores=scores,
                attentions=None,
                hidden_states=None
            )
        else:
            return generated_sequences

    def forward(
            self,
            input_ids: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            labels: Optional[torch.Tensor] = None,
            **kwargs
    ):
        """Mock forward pass"""
        batch_size, seq_len = input_ids.shape

        # Generate random logits
        logits = torch.randn(
            batch_size, seq_len, self.config.vocab_size,
            device=input_ids.device
        )

        # Calculate loss if labels provided
        loss = None
        if labels is not None:
            loss = torch.randn(1, device=input_ids.device)

        # Return a simple namespace object with common attributes
        class Output:
            pass

        output = Output()
        output.loss = loss
        output.logits = logits

        return output

    def __call__(self, *args, **kwargs):
        """Make the model callable"""
        return self.forward(*args, **kwargs)

    def parameters(self):
        """Mock parameters method like nn.Module"""
        # Create some dummy parameters to mimic a real model
        dummy_params = [
            torch.nn.Parameter(torch.randn(self.config.vocab_size, self.config.hidden_size)),
            torch.nn.Parameter(torch.randn(self.config.hidden_size)),
            torch.nn.Parameter(torch.randn(self.config.hidden_size, self.config.hidden_size)),
        ]
        return iter(dummy_params)


# Example usage:
if __name__ == "__main__":
    # Create mock model
    model = MockAutoModelForCausalLM()

    # Test config access
    print(f"Model name: {model.config.name_or_path}")
    print(f"Vocab size: {model.config.vocab_size}")

    # Create mock inputs
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 2
    seq_len = 10


    # Mock inputs object
    class MockInputs:
        def __init__(self, input_ids, attention_mask):
            self.input_ids = input_ids
            self.attention_mask = attention_mask


    input_ids = torch.randint(0, model.config.vocab_size, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len)
    inputs = MockInputs(input_ids, attention_mask)

    # Test generate method
    output = model.generate(
        inputs.input_ids.to(device),
        attention_mask=inputs.attention_mask.to(device),
        max_new_tokens=20,
        do_sample=False,
        pad_token_id=50256,
        return_dict_in_generate=True,
        output_scores=True,
    )

    print(f"Generated sequences shape: {output.sequences.shape}")
    print(f"Number of score tensors: {len(output.scores) if output.scores else 0}")
