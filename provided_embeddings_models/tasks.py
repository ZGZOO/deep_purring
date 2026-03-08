import dataclasses
from typing import Literal, Dict

TaskType = Literal["age_group", "age", "gender"]


@dataclasses.dataclass(frozen=True)
class TaskConfig:
    """
    Task configuration
    - param output_size (int): number of classes to predict (1 for regression)
    - param regression (bool): True if this is a regression task, False otherwise
    - param direction: 'maximize' or 'minimize'
    """
    output_size: int
    regression: bool
    direction: Literal["maximize", "minimize"]


TASK_CONFIGS: Dict[TaskType, TaskConfig] = {
    "age_group": TaskConfig(3, False, "maximize"),
    "age": TaskConfig(1, True, "minimize"),
    "gender": TaskConfig(2, False, "maximize"),
}
