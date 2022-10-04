import unittest

import numpy as np

from exqaliber.sampling_schedule.fixed_sampling_schedule.base import (
    SAMPLING_SCHEDULE, BaseSamplingSchedule)


class LinearIncrementalSequence(BaseSamplingSchedule):
    '''Fixed sampling method class

    Attributes
    ----------
    __type: SAMPLING_SCHEDULE.LINEAR_INCREMENTAL_SEQUENCE
        Linear incremental sequence sampling schedule
    __n_shots : int
        Number of circuit shots for each m
    __seq_length : int
        Final m value

    Methods
    -------
    get_n_shots()
        Get number of shots for each m
    get_seq_length()
        Get final m value
    get_sampling_schedule()
        Get the linear incremental sequence pairs
    '''

    def __init__(self, n_shots: int, seq_length: int):
        '''
        Parameters
        ----------
        __n_shots : int
            Number of circuit shots for each m
        __seq_length : int
            Final m value
        '''

        super().__init__(SAMPLING_SCHEDULE.LINEAR_INCREMENTAL_SEQUENCE)
        self.__n_shots = n_shots
        self.__seq_length = seq_length

    def get_n_shots(self) -> int:
        '''Get number of shots for each m

        Returns
        -------
        int :
            Number of circuit shots for each m
        '''

        return self.__n_shots

    def get_seq_length(self) -> int:
        '''Get final m value

        Returns
        -------
        int :
            Final m value
        '''

        return self.__seq_length

    def get_sampling_schedule(self) -> np.ndarray:
        '''Get the linear incremental sequence pairs

        Returns
        -------
        np.ndarray((self.__seq_length + 1), 2) :
            Pairs of (n_shots, m) for m = 0, ..., seq_length

        '''

        return np.vstack((np.arange(self.__seq_length + 1), self.__n_shots * np.ones(self.__seq_length + 1))).transpose()
