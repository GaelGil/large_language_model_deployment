from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Config(BaseSettings):
    model_config = SettingsConfigDict(env_file="../.env", env_file_encoding="utf-8")

    TOKENIZER_PATH: str = Field(default="tokenizer/model/joint.model")
    MODEL_CHECKPOINT_PATH: Path = Field(default=Path("tokenizer/model/joint.model"))
    SEQ_LEN: int = Field(default=256)
    MAX_LEN: int = Field(default=128)
    EOS_ID: int = Field(default=2)
    BOS_ID: int = Field(default=1)

    VOCAB_SIZE: int = Field(default=8000)
    NUM_LAYERS: int = Field(default=6)
    HIDDEN_DIM: int = Field(default=512)
    NUM_HEADS: int = Field(default=8)
    EXPANDED_FEED_FORWARD: int = Field(default=2048)
    DROPOUT_RATE: float = Field(default=0.1)

    MODAL_VOLUME_NAME: str = Field(default="seq2seq-model-weights")
    MODAL_GPU: str = Field(default="T4")
    MODAL_MEMORY: int = Field(default=8192)


CONFIG = Config()
