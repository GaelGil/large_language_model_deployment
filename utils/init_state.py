import orbax.checkpoint as ocp
from flax import nnx
from transformer.Transformer import Transformer


def init_state(
    src_vocab_size: int,
    target_vocab_size: int,
    D_MODEL: int,
    N: int,
    H: int,
    D_FF: int,
    SEQ_LEN: int,
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
            d_model=D_MODEL,
            N=N,
            n_heads=H,
            d_ff=D_FF,
            dropout=None,
            seq_len=SEQ_LEN,
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
        d_model=D_MODEL,
        N=N,
        n_heads=H,
        d_ff=D_FF,
        dropout=None,
        seq_len=SEQ_LEN,
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
