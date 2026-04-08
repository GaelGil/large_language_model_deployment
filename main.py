from pathlib import Path
import sentencepiece as spm
import modal
from config import CONFIG
app = modal.App("seq2seq-jax-inference")

image = modal.Image.debian_slim().uv_pip_install(
    "jax[gpu]",
    "flax",
    "numpy",
    "orbax-checkpoint",
    "sentencepiece",  # if you use it
)

model_volume = modal.Volume.from_name("seq2seq-model-weights", create_if_missing=True)
MODEL_DIR = "/models"


@app.cls(
    image=image,
    volumes={MODEL_DIR: model_volume},
    cpu=4.0,
    memory=8192,
)
class Seq2SeqModel:
    @modal.enter()
    def load(self):
        self.model_path = Path(MODEL_DIR) / "my_checkpoint"

        self.tokenizer = spm.SentencePieceProcessor().Load(f"{CONFIG.SRC_CORPUS_PATH}/{config.model_prefix}.model")config.SEQ_LEN,)



    @modal.method()
    def predict(self, text: str, max_new_tokens: int = 128) -> dict:
        tokens = self.tokenizer.encode(text)
        output = run_inference(self.model, self.params, tokens, max_new_tokens)
        decoded = self.tokenizer.decode(output)
        decoded = f"stub-output-for: {text}"
        return {"input": text, "output": decoded}
