"""Implementation of a linear sampling schedule."""
import numpy as np

from exqaliber.sampling_schedule.fixed_sampling_schedule.base import (
    SAMPLING_SCHEDULE,
    BaseSamplingSchedule,
)


class LinearIncrementalSequence(BaseSamplingSchedule):
    """Fixed sampling method class.

    Attributes
    ----------
    __type: SAMPLING_SCHEDULE.LINEAR_INCREMENTAL_SEQUENCE
        Linear incremental sequence sampling schedule
    __n_shots : int
        Number of circuit shots for each m
    __seq_length : int
        Final m value

    """

    def __init__(self, n_shots: int, seq_length: int):
        """Initialise LinearIncrementalSequece.

        Parameters
        ----------
        __n_shots : int
            Number of circuit shots for each m
        __seq_length : int
            Final m value
        """
        super().__init__(SAMPLING_SCHEDULE.LINEAR_INCREMENTAL_SEQUENCE)
        self.__n_shots = n_shots
        self.__seq_length = seq_length

    def get_n_shots(self) -> int:
        """Get number of shots for each m.

        Returns
        -------
        int :
            Number of circuit shots for each m
        """
        return self.__n_shots

    def get_seq_length(self) -> int:
        """Get final m value.

        Returns
        -------
        int :
            Final m value
        """
        return self.__seq_length

    def get_n_shots_schedule(self) -> np.ndarray:
        """Get the schedule for numbers of shots (n_0, n_1, ...).

        Returns
        -------
        np.ndarray :
            Schedule for number of shots at each Grover depth
        """
        return self.__n_shots * np.ones(self.__seq_length + 1).astype(int)

    def get_grover_depth_schedule(self) -> list[tuple[int, int]]:
        """Get the schedule for Grover depth (m_0, m_1, ...).

        Returns
        -------
        np.ndarray((self.__seq_length + 1), 2) :
            Schedule for Grover depth,
            (0,1,2,3,...,`self.__seq_length`-1)
        """
        return np.arange(self.__seq_length + 1).astype(int)
