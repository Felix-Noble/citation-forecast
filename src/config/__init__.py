from . import cuda, data, env, logging, model, train

__all__ = [
    "env",
    "data",
    "cuda",
    "logging",
    "model",
    "train",
]
assert env.STAGED_LOC.exists(), "Staged loc must exist"

assert (env.STAGED_LOC / data.train.dataset.loc).exists(), "train loc must exist"
assert (env.STAGED_LOC / data.test.dataset.loc).exists(), "test loc must exist"
