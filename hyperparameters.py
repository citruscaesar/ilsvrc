from dataclasses import dataclass, asdict
from typing import Callable

@dataclass(frozen = True, repr = True)
class Hyperparameters:
    task: str
    random_seed: int
    num_classes: int
    metrics: list[str]

    criterion: Callable | str
    optimizer: Callable | str  
    learning_rate: float
    momentum: float
    weight_decay: float

    batch_size: int
    grad_accum: int
    test_split: float
    transform : list[str]

    num_workers: int

    def get_dict(self) -> dict:
        return asdict(self)

    def get_datamodule_dict(self) -> dict:    
        return {
            "batch_size": self.batch_size // self.grad_accum,
            "transform": self.transform,
            "test_split": self.test_split,

            "num_workers": self.num_workers,
        }

    def get_litmodule_hparams(self) -> dict:
        return {
            "task": self.task,
            "num_classes": self.num_classes,
            "metrics": self.metrics,

            "criterion": self.criterion,
            "optimizer": self.optimizer,
            "learning_rate": self.learning_rate,
            "momentum": self.momentum,
            "weight_decay": self.weight_decay,
        }