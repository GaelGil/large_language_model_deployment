from pydantic_settings import BaseSettings  # handles .env


class Config(BaseSettings):
    SRC_CORPUS_PATH: str = ""
    TOKENIZER_PATH: str = "tokenizer/model/joint.model"  # path inside volume
    SEQ_LEN: int = 256
    MODEL_PATH: str = "tokenizer/model/joint.model"  # checkpoint path
    MAX_LEN: int = 128
    EOS_ID: int = 2  # sentencepiece eos id

    class Config:
        env_file = ".env"
        case_sensitive = False
