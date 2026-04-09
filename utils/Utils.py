import jax.numpy as jnp
from jax import Array


class Utils:
    def __init__(self):
        pass

    def decode(self, tokenizer, ids: list[int]) -> str:
        """Decode token IDs to text using SentencePiece."""
        return tokenizer.Decode(ids)

    def _create_causal_mask(self, seq_len: int) -> Array:
        """
        Create causal (triangular) attention mask.

        This ensures each token can only attend to previous tokens,
        which is required for autoregressive generation.
        """
        mask = jnp.tril(jnp.ones((seq_len, seq_len), dtype=jnp.bool_))[None, None, :, :]
        return mask

    def encode(
        self,
        text: str,
        tokenizer,
        max_len: int,
        add_bos: bool = False,
        add_eos: bool = False,
        prefix: str = "",
    ) -> list[int]:
        """
        Encode text to token IDs using SentencePiece.

        Args:
            text: Input text to encode
            add_bos: Prepend BOS token
            add_eos: Append EOS token
            prefix: Optional prefix to prepend (e.g., "<es_to_en>")

        Returns:
            List of token IDs
        """
        full_text = prefix + text if prefix else text
        ids = tokenizer.Encode(full_text, out_type=int)

        # Truncate if needed
        if max_len is not None:
            reserved = (1 if add_bos else 0) + (1 if add_eos else 0)
            content_max = max_len - reserved
            if content_max < 0:
                raise ValueError("max_len too small for requested special tokens")
            ids = ids[:content_max]

            # Add special tokens
            if add_bos:
                ids = [self.bos_id] + ids
            if add_eos:
                ids = ids + [self.eos_id]

        return ids
