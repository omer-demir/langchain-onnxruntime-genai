from enum import Enum


class ExecutionProviders(Enum):
    CPU = "cpu"
    CUDA = "cuda"
    DML = "dml"
