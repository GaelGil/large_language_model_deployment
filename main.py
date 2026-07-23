import json
from typing import Generator

import jax.numpy as jnp
import modal
import orbax.checkpoint as ocp
from fastapi import Depends
from fastapi.responses import StreamingResponse
from flax import nnx
from sentencepiece import SentencePieceProcessor

from utils.config import CONFIG
from utils.Utils import TranslationRequest, Utils, require_auth

# define the app name
app = modal.App("seq2seq-translator")

# define the container image
image = (
    modal.Image.debian_slim()
    .uv_pip_install(
        "jax[cuda13]",
        "flax",
        "numpy",
        "sentencepiece",
        "orbax-checkpoint",
        "pydantic",
        "pydantic-settings",
        "fastapi[standard]",
    )
    .add_local_python_source("transformer", "utils")
    .add_local_file(
        "tokenizer/model/joint.model",
        remote_path=str(CONFIG.TOKENIZER_PATH),
    )
)

# load the persistent Modal volume (storage) that stores model checkpoints.
checkpoint_volume = modal.Volume.from_name(CONFIG.MODAL_VOLUME_NAME)


# create a class
@app.cls(
    image=image,
    volumes={
        "/model": checkpoint_volume
    },  # attach the checkpoint volume and show as /model
    gpu=CONFIG.MODAL_GPU,
    memory=CONFIG.MODAL_MEMORY,
    secrets=[modal.Secret.from_name("translator-auth-token")],
)
class Translator:
    @modal.enter()
    def load(self):
        """Load tokenizer and model on container startup."""
        # -------------------------------------------------------------------------
        #  load SentencePiece tokenizer
        # -------------------------------------------------------------------------
        self.sp = SentencePieceProcessor()
        self.sp.Load(str(CONFIG.TOKENIZER_PATH))
        self.utils = Utils()

        # -------------------------------------------------------------------------
        # initialize the checkpoint manager
        # -------------------------------------------------------------------------
        manager = ocp.CheckpointManager(
            directory=CONFIG.MODEL_CHECKPOINT_DIR.resolve(),
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

    def stream_token_ids(
        self,
        src_text: str,
        max_new_tokens: int = 128,
    ) -> Generator[int, None, None]:
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
                "Model not loaded. Please set MODEL_CHECKPOINT_DIR and implement "
                "checkpoint loading in the load() method."
            )

        eos_id = self.sp.eos_id()
        bos_id = self.sp.bos_id()

        if eos_id < 0:
            raise RuntimeError("Tokenizer does not define an EOS token")

        if bos_id < 0:
            raise RuntimeError("Tokenizer does not define an BOS token")

        # -------------------------------------------------------------------------
        # encode source text
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
        # initialize decoder with BOS
        # -------------------------------------------------------------------------
        en_ids = [bos_id]
        en = jnp.array([en_ids], dtype=jnp.int32)  # [1, tgt_len]

        # Empty KV cache
        self_attention_cache = None

        past_len = 0
        decoder_mask = self.utils._create_causal_mask(
            current_len=en.shape[1], past_len=past_len
        )

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
        yield next_token

        # Append to decoder input for next iteration
        en = jnp.array([[next_token]], dtype=jnp.int32)

        # -------------------------------------------------------------------------
        # autoregressive generation loop
        # -------------------------------------------------------------------------
        for _ in range(max_new_tokens - 1):
            # Create causal mask for current sequence length
            past_len = (
                self_attention_cache[0][0].shape[2]
                if self_attention_cache is not None
                else 0
            )
            decoder_mask = self.utils._create_causal_mask(
                current_len=en.shape[1], past_len=past_len
            )
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
            # Take argmax of last token logits
            next_token = int(jnp.argmax(logits[0, -1]))

            # Check for EOS
            if next_token == eos_id:
                break

            # Yield token ID for streaming
            yield next_token

            # Append to decoder input for next iteration
            en = jnp.array([[next_token]], dtype=jnp.int32)

    @modal.fastapi_endpoint(method="POST")
    def translate(
        self, request: TranslationRequest, _: None = Depends(require_auth)
    ) -> StreamingResponse:
        def event_stream() -> Generator[bytes, None, None]:
            generated_ids: list[int] = []
            output_text = ""
            # for every token id
            for token_id in self.stream_token_ids(request.text, request.max_new_tokens):
                # add it to generated ids
                generated_ids.append(int(token_id))
                # decode the ids
                decoded_text = self.sp.Decode(generated_ids)
                # return the most recent one
                delta = decoded_text[len(output_text) :]
                # stream it
                if delta:
                    yield f"data: {json.dumps({'text': delta})}\n\n".encode()
                    output_text = decoded_text
            yield b"event: done\ndata: {}\n\n"

        return StreamingResponse(event_stream(), media_type="text/event-stream")
