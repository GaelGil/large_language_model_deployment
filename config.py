from pydantic import BaseModel


class Config(BaseModel):
    SRC_CORPUS_PATH: str
    TOKENIZER_PATH: str
    SEQ_LEN: int
    MODEL_PATH: str
    MAX_LEN: int


CONFIG = Config(
    SRC_CORPUS_PATH="", TOKENIZER_PATH="", SEQ_LEN=1, MODEL_PATH="", MAX_LEN=1
)
