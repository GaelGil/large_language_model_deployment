"""
Modal deployment for JAX NNX translation model.

Usage:
    modal deploy main.py

Local testing:
    modal run main --local
"""

from typing import Generator

import jax.numpy as jnp
import modal
import orbax.checkpoint as ocp
from sentencepiece import SentencePieceProcessor

from utils.config import CONFIG
from utils.Utils import Utils

app = modal.App("seq2seq-translator")

image = modal.Image.debian_slim().uv_pip_install(
    "jax[cuda12]",
    "flax",
    "numpy",
    "sentencepiece",
)

model_volume = modal.Volume.from_name(
    CONFIG.MODAL_VOLUME_NAME,
    create_if_missing=True,
)


utils = Utils()


@app.cls(
    image=image,
    volumes={"/model": model_volume},
    gpu=CONFIG.MODAL_GPU,
    memory=CONFIG.MODAL_MEMORY,
)
class Translator:
    @modal.enter()
    def load(self):
        """Load tokenizer and model on container startup."""
        # -------------------------------------------------------------------------
        # 1. Load SentencePiece tokenizer
        # -------------------------------------------------------------------------
        self.sp = SentencePieceProcessor()
        self.sp.Load("/model/joint.model")
        self.eos_id = self.sp.eos_id()
        self.bos_id = self.sp.bos_id()

        # initialize the checkpoint manager with the options

        manager = ocp.CheckpointManager(
            directory=CONFIG.MODEL_CHECKPOINT_PATH.resolve(),
        )

        # Placeholder - replace with actual model loading
        self.model = None  # Replace with loaded model
        self._infer_fn = None  # Replace with JIT-compiled inference function

    @modal.method()
    def stream_translation(
        self,
        src_text: str,
        max_new_tokens: int = 128,
    ) -> Generator[str, None, None]:
        """
        Translate text with token-by-token streaming.

        Args:
            src_text: Source text to translate
            max_new_tokens: Maximum tokens to generate

        Yields:
            Token IDs as strings, one at a time
        """
        if self.model is None:
            raise RuntimeError(
                "Model not loaded. Please set MODEL_CHECKPOINT_PATH and implement "
                "checkpoint loading in the load() method."
            )

        # -------------------------------------------------------------------------
        # 1. Encode source text
        # -------------------------------------------------------------------------
        es_ids = self.utils.encode(src_text, add_bos=False, add_eos=False)
        es = jnp.array([es_ids], dtype=jnp.int32)  # [1, src_len]

        # -------------------------------------------------------------------------
        # 2. Initialize decoder with BOS
        # -------------------------------------------------------------------------
        en_ids = [self.bos_id]
        en = jnp.array([en_ids], dtype=jnp.int32)  # [1, tgt_len]

        # -------------------------------------------------------------------------
        # 3. Autoregressive generation loop
        # -------------------------------------------------------------------------
        for _ in range(max_new_tokens):
            # Create causal mask for current sequence length
            decoder_mask = self._create_causal_mask(en.shape[1])

            # Forward pass
            logits = self._infer_fn(
                src=es,
                target=en,
                src_mask=None,
                self_mask=decoder_mask,
                cross_mask=None,
                is_training=False,
            )

            # Greedy sampling: take argmax of last token logits
            next_token = int(jnp.argmax(logits[0, -1]))

            # Check for EOS
            if next_token == self.eos_id:
                break

            # Yield token ID for streaming
            yield str(next_token)

            # Append to decoder input for next iteration
            en_ids.append(next_token)
            en = jnp.array([en_ids], dtype=jnp.int32)


# =============================================================================
# Local development / testing
# =============================================================================


@app.local_entrypoint()
def main():
    """Local entry point for testing."""
    translator = Translator()
    translator.load()

    encode = utils.encode()

    print("\n--- Streaming mode ---")
    for token_id in translator.translate_streaming(test_text):
        token = translator.decode([int(token_id)])
        print(f"Token {token_id}: '{token}'")
