import modal
import sentencepiece as spm

from config import CONFIG

app = modal.App("seq2seq-jax-inference")

image = modal.Image.debian_slim().uv_pip_install(
    "jax[gpu]",
    "flax",
    "numpy",
    "orbax-checkpoint",
    "sentencepiece",
)

model_volume = modal.Volume.from_name("seq2seq-model-weights", create_if_missing=True)


@app.cls(
    image=image,
    volumes={CONFIG.MODEL_PATH: model_volume},
    cpu=4.0,
    memory=8192,
)
class Seq2SeqModel:
    @modal.enter()
    def load(self):
        self.model_path = CONFIG.MODEL_PATH
        sp = spm.SentencePieceProcessor()
        self.tokenizer = sp.Load(CONFIG.MODEL_PATH)

    @modal.method()
    def predict(self, text: str, max_new_tokens: int = 128) -> dict:
        tokens = self.tokenizer.encode(text)
        output = run_inference(self.model, self.params, tokens, max_new_tokens)
        decoded = self.tokenizer.decode(output)
        decoded = f"stub-output-for: {text}"
        return {"input": text, "output": decoded}
