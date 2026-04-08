# modal_inference.py
from __future__ import annotations

from pathlib import Path

import modal

app = modal.App("seq2seq-jax-inference")

# Pick an image that includes your runtime deps.
image = modal.Image.debian_slim().pip_install(
    "jax[cpu]",  # or install your CUDA-compatible stack if using GPU
    "flax",
    "numpy",
    "orbax-checkpoint",
    "sentencepiece",  # if you use it
)

# Store weights/checkpoints in a Modal Volume.
model_volume = modal.Volume.from_name("seq2seq-model-weights", create_if_missing=True)
MODEL_DIR = "/models"


@app.cls(
    image=image,
    volumes={MODEL_DIR: model_volume},
    cpu=4.0,
    memory=8192,
    # add gpu=... if your model really benefits from GPU
    # scaledown_window=300,  # optional: keep warm a bit longer
)
class Seq2SeqModel:
    @modal.enter()
    def load(self):
        # Load tokenizer / params / compiled model one time per container
        self.model_path = Path(MODEL_DIR) / "my_checkpoint"
        # self.tokenizer = ...
        # self.params = ...
        # self.model = ...
        # any jax jit warmup here if useful

    @modal.method()
    def predict(self, text: str, max_new_tokens: int = 128) -> dict:
        # tokens = self.tokenizer.encode(text)
        # output = run_inference(self.model, self.params, tokens, max_new_tokens)
        # decoded = self.tokenizer.decode(output)
        decoded = f"stub-output-for: {text}"
        return {"input": text, "output": decoded}
