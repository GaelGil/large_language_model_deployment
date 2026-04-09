import jax.numpy as jnp
import orbax.checkpoint as ocp
from flax import nnx
from jax import Array

from transformer.Transformer import Transformer


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
        bos_id: int,
        eos_id: int,
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
                ids = [bos_id] + ids
            if add_eos:
                ids = ids + [eos_id]

        return ids

    def init_state(
        self,
        src_vocab_size: int,
        target_vocab_size: int,
        D_MODEL: int,
        N: int,
        H: int,
        D_FF: int,
        SEQ_LEN: int,
        manager: ocp.CheckpointManager,
    ) -> Transformer:
        """
        Initialize the state from a checkpoint or create a new one
        Args:
            config: Config

        Returns:
        tuple[Transformer, nnx.Optimizer]
        """

        # create abstract model
        abs_model = nnx.eval_shape(
            lambda: Transformer(
                d_model=D_MODEL,
                N=N,
                n_heads=H,
                d_ff=D_FF,
                dropout=0,
                seq_len=SEQ_LEN,
                src_vocab_size=src_vocab_size,
                target_vocab_size=target_vocab_size,
                rngs=nnx.Rngs(0),
            )
        )

        # split the abstract model into graphdef, state and rng
        abs_state = nnx.state(abs_model)

        # create model
        rngs = nnx.Rngs(0)
        model: Transformer = Transformer(
            d_model=D_MODEL,
            N=N,
            n_heads=H,
            d_ff=D_FF,
            dropout=0,
            seq_len=SEQ_LEN,
            src_vocab_size=src_vocab_size,
            target_vocab_size=target_vocab_size,
            rngs=rngs,
        )

        # restore the model
        restored = manager.restore(
            step=manager.latest_step(),
            args=ocp.args.Composite(
                state=ocp.args.StandardRestore(abs_state),
            ),
        )
        nnx.update(model, restored["state"])
        return model
