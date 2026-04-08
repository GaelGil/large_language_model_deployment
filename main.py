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
        self.model = .load(CONFIG.MODEL_PATH)
        sp = spm.SentencePieceProcessor()
        self.tokenizer = sp.Load(CONFIG.MODEL_PATH)

    def encode(self, text: str, add_bos: bool, add_eos: bool):
        ids = self.tokenizer.Encode(text, out_type=int)

        if CONFIG.MAX_LEN is not None:
            reserved = (1 if add_bos else 0) + (1 if add_eos else 0)
            content_max = CONFIG.MAX_LEN - reserved
            if content_max < 0:
                raise ValueError("max_len too small for requested special tokens")
            ids = ids[:content_max]

        if add_bos:
            ids = [self.tokenizer.bos_id()] + ids
        if add_eos:
            ids = ids + [self.tokenizer.eos_id()]

        return ids

    @modal.method()
    def run_inference(self, src, self_mask):
        logits = self.model(
            src=src,
            target=self.encode(text="", add_bos=True, add_eos=False),
            src_mask=None,
            self_mask=self_mask,
            cross_mask=None,
            is_training=False,
        )

    @modal.method()
    def predict(self, text: str, max_new_tokens: int = 128) -> dict:
        tokens = self.encode(text, add_bos=True, add_eos=True)
        output = run_inference(self.model, self.params, tokens, max_new_tokens)
        decoded = self.tokenizer.decode(output)
        decoded = f"stub-output-for: {text}"
        return {"input": text, "output": decoded}
