from typing import Generator

import jax.numpy as jnp
import modal
import orbax.checkpoint as ocp
from flax import nnx
from sentencepiece import SentencePieceProcessor

from utils.config import CONFIG
from utils.Utils import Utils

app = modal.App("seq2seq-translator")

image = modal.Image.debian_slim().uv_pip_install(
    "jax[cuda13]",
    "flax",
    "numpy",
    "sentencepiece",
    "orbax-checkpoint",
    "pydantic",
    "pydantic-settings",
)

model_volume = modal.Volume.from_name(
    CONFIG.MODAL_VOLUME_NAME,
    create_if_missing=True,
)


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
        #  Load SentencePiece tokenizer
        # -------------------------------------------------------------------------
        self.sp = SentencePieceProcessor()
        self.sp.Load(CONFIG.TOKENIZER_PATH)
        self.utils = Utils()

        # -------------------------------------------------------------------------
        # initialize the checkpoint manager
        # -------------------------------------------------------------------------
        manager = ocp.CheckpointManager(
            directory=CONFIG.MODEL_CHECKPOINT_PATH.resolve(),
        )

        # -------------------------------------------------------------------------
        # restore the model
        # -------------------------------------------------------------------------
        self.model = self.utils.init_state(
            src_vocab_size=CONFIG.VOCAB_SIZE,
            target_vocab_size=CONFIG.VOCAB_SIZE,
            D_MODEL=CONFIG.D_MODEL,
            N=CONFIG.N,
            H=CONFIG.H,
            D_FF=CONFIG.D_FF,
            SEQ_LEN=CONFIG.SEQ_LEN,
            manager=manager,
        )
        self._infer_fn = nnx.jit(
            self.model, static_argnames=["is_training", "use_cache"]
        )

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

        eos_id = self.sp.eos_id()
        bos_id = self.sp.bos_id()

        if eos_id < 0:
            raise RuntimeError("Tokenizer does not define an EOS token")

        if bos_id < 0:
            raise RuntimeError("Tokenizer does not define an BOS token")

        # -------------------------------------------------------------------------
        # 1. Encode source text
        # -------------------------------------------------------------------------
        es_ids = self.utils.encode(
            src_text,
            add_bos=False,
            add_eos=False,
            eos_id=eos_id,
            bos_id=bos_id,
            tokenizer=self.sp,
            max_len=CONFIG.MAX_LEN,
        )
        es = jnp.array([es_ids], dtype=jnp.int32)  # [1, src_len]

        # -------------------------------------------------------------------------
        # 2. Initialize decoder with BOS
        # -------------------------------------------------------------------------
        en_ids = [bos_id]
        en = jnp.array([en_ids], dtype=jnp.int32)  # [1, tgt_len]

        # Empty KV cache
        self_attention_cache = [None] * CONFIG.N

        decoder_mask = self.utils._create_causal_mask(en.shape[1])

        # Forward pass
        logits, cache = self._infer_fn(
            src=es,
            target=en,
            src_mask=None,
            self_mask=decoder_mask,
            cross_mask=None,
            is_training=False,
            self_attention_cache=self_attention_cache,
            use_cache=True,
        )

        encoder_output = cache["encoder_output"]
        self_attention_cache = cache["self_attention_cache"]

        next_token = int(jnp.argmax(logits[0, -1]))

        # Check for EOS
        if next_token == eos_id:
            return

        # Yield token ID for streaming
        yield str(next_token)

        # Append to decoder input for next iteration
        en_ids.append(next_token)

        # -------------------------------------------------------------------------
        # 3. Autoregressive generation loop
        # -------------------------------------------------------------------------
        for _ in range(max_new_tokens):
            # Create causal mask for current sequence length
            decoder_mask = self.utils._create_causal_mask(en.shape[1])

            # Forward pass
            logits, cache = self._infer_fn(
                src=es,
                target=en,
                src_mask=None,
                self_mask=decoder_mask,
                cross_mask=None,
                is_training=False,
                self_attention_cache=self_attention_cache,
                encoder_output=encoder_output,
                use_cache=True,
            )

            self_attention_cache = cache["self_attention_cache"]
            # Greedy sampling: take argmax of last token logits
            next_token = int(jnp.argmax(logits[0, -1]))

            # Check for EOS
            if next_token == eos_id:
                break

            # Yield token ID for streaming
            yield str(next_token)

            # Append to decoder input for next iteration
            en_ids.append(next_token)
            en = jnp.array([en_ids], dtype=jnp.int32)


# we present humanlike ai for attending meetings, sales calls.
# Send your own personal AI avatar to atteend meetings, take notes, ask questions.
