from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class Seru:
    workers_set: List[int] = field(default_factory=list)
    batches_set: List[int] = field(default_factory=list)
    throughput_time: float = 0.0  # 保持不变 (用于计算拖期)
    processing_time: float = 0.0  # Used to calculate makespan.
    labour_time: float = 0.0
    tardiness: float = 0.0
    fitness: float = 0.0

    def __copy__(self):
        # 实现深拷贝
        return Seru(
            workers_set=self.workers_set.copy(),
            batches_set=self.batches_set.copy(),
            throughput_time=self.throughput_time,
            processing_time=self.processing_time,
            labour_time=self.labour_time,
            tardiness=self.tardiness,
            fitness=self.fitness
        )


@dataclass
class SeruSchedule:
    batches_throughput_time_in_seru: List[Tuple[int, float]] = field(default_factory=list)
    schedule_code: List[int] = field(default_factory=list)
    batches_assignment: List[List[int]] = field(default_factory=list)
    makespan: float = 0.0
    labour_time: float = 0.0
    tardiness: float = 0.0
    fitness: float = 0.0

    def __copy__(self):
        return SeruSchedule(
            batches_throughput_time_in_seru=[(entry[0], entry[1]) for entry in self.batches_throughput_time_in_seru],
            schedule_code=self.schedule_code.copy(),
            batches_assignment=[inner.copy() for inner in self.batches_assignment],
            makespan=self.makespan,
            labour_time=self.labour_time,
            tardiness=self.tardiness,
            fitness=self.fitness
        )


@dataclass
class SeruFormation:
    seru_set: List[Seru] = field(default_factory=list)
    formation_code: List[int] = field(default_factory=list)
    makespan: float = 0.0
    labour_time: float = 0.0
    tardiness: float = 0.0
    fitness: float = 0.0

    def __copy__(self):
        return SeruFormation(
            seru_set=[seru.__copy__() for seru in self.seru_set],
            formation_code=self.formation_code.copy(),
            makespan=self.makespan,
            labour_time=self.labour_time,
            tardiness=self.tardiness,
            fitness=self.fitness
        )


@dataclass
class Solution:
    # 解的 makespan
    makespan: float = 0.0  # Completion-time objective.

    # 解的 labour_time
    labour_time: float = 0.0

    # 解的 tardiness
    tardiness: float = 0.0

    # 解的 fitness
    fitness: float = 0.0

    # 当前解的 seru 构造
    formation: 'SeruFormation' = field(default_factory=lambda: SeruFormation())

    # 当前解的 seru 调度
    schedule: 'SeruSchedule' = field(default_factory=lambda: SeruSchedule())

    def __copy__(self):
        # 深拷贝
        return Solution(
            makespan=self.makespan,
            labour_time=self.labour_time,
            tardiness=self.tardiness,
            fitness=self.fitness,
            formation=self.formation.__copy__(),
            schedule=self.schedule.__copy__()
        )
