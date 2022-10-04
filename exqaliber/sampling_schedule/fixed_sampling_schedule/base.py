"""Base class and Enum type for sampling schedules."""
from abc import ABC, abstractmethod
from enum import Enum

import numpy as np


class SAMPLING_SCHEDULE(Enum):
    r"""Type of sampling schedule as an Enum."""

    LINEAR_INCREMENTAL_SEQUENCE = 0
    EXPONENTIAL_INCREMENTAL_SEQUENCE = 1


class BaseSamplingSchedule(ABC):
    """Abstract base class for sampling schedules.

    Attributes
    ----------
    __type : SAMPLING_SCHEDULE
        Sampling schedule scheme

    """

    def __init__(self, sampling_schedule: SAMPLING_SCHEDULE):
        """Initialise BaseSamplingSchedule.

        Parameters
        ----------
        sampling_schedule : SAMPLING_SCHEDULE
            Sampling schedule scheme
        """
        self.__type = sampling_schedule

    def get_type(self) -> SAMPLING_SCHEDULE:
        """Get the type of the sampling schedule.

        Returns
        -------
        SAMPLING_SCHEDULE:
            type of the sampling schedule
        """
        return self.__type

    @abstractmethod
    def get_sampling_schedule(self) -> np.ndarray:
        """Get the sampling schedule pairs (m_i, n_shots_i)."""
        pass
