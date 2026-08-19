from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Config(BaseSettings):
    model_config = SettingsConfigDict(env_file="../.env", env_file_encoding="utf-8")

    TOKENIZER_PATH: str = Field(default="/tokenizer/model/joint.model")
    MODEL_CHECKPOINT_DIR: Path = Field(default=Path("/model"))
    MODEL_CHECKPOINTS: dict[str, str] = {
        "english": "english",
        "spanish": "spanish",
    }
    SEQ_LEN: int = Field(default=256)
    EOS_ID: int = Field(default=2)
    BOS_ID: int = Field(default=1)
    D_MODEL: int = Field(default=512)
    N: int = Field(default=6)
    H: int = Field(default=8)
    D_FF: int = Field(default=2048)

    VOCAB_SIZE: int = Field(default=8000)

    MODAL_VOLUME_NAME: str = Field(default="model-checkpoints")
    MODAL_GPU: str = Field(default="T4")
    MODAL_MEMORY: int = Field(default=8192)


CONFIG = Config()
