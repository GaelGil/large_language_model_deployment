import orbax.checkpoint as ocp
from flax import nnx
from transformer.Transformer import Transformer

from utils.config import Config


def init_state(
    config: Config,
    src_vocab_size: int,
    target_vocab_size: int,
    manager: ocp.CheckpointManager,
) -> tuple[Transformer, nnx.Optimizer, int]:
    """
    Initialize the state from a checkpoint or create a new one
    Args:
        config: Config

    Returns:
       tuple[Transformer, nnx.Optimizer]
    """

    # create abstract model
    abs_model = nnx.eval_shape(
        lambda: Transformer(
            d_model=config.D_MODEL,
            N=config.N,
            n_heads=config.H,
            d_ff=config.D_FF,
            dropout=config.DROPOUT,
            seq_len=config.SEQ_LEN,
            src_vocab_size=src_vocab_size,
            target_vocab_size=target_vocab_size,
            rngs=nnx.Rngs(0),
        )
    )

    # split the abstract model into graphdef, state and rng
    abs_state = nnx.state(abs_model)

    # create model
    rngs = nnx.Rngs(0)
    model: Transformer = Transformer(
        d_model=config.D_MODEL,
        N=config.N,
        n_heads=config.H,
        d_ff=config.D_FF,
        dropout=config.DROPOUT,
        seq_len=config.SEQ_LEN,
        src_vocab_size=src_vocab_size,
        target_vocab_size=target_vocab_size,
        rngs=rngs,
    )

    restored = manager.restore(
        step=manager.latest_step(),
        args=ocp.args.Composite(
            state=ocp.args.StandardRestore(abs_state),
        ),
    )
    nnx.update(model, restored["state"])
    # return the restored model and optimizer
    return model
