import numpy as np
import unittest

from src.circuit_models.fixed_sampling_schedule.base import BaseSamplingSchedule, SAMPLING_SCHEDULE

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


class LinearIncrementalSequenceTestCase(unittest.TestCase):
    def setUp(self):
        self.n_shots = 100
        self.max_m = 10
        self.sampling_schedule = LinearIncrementalSequence(self.n_shots,self.max_m)
    
    def test_sequence_length(self):
        self.assertEqual(self.max_m + 1, self.sampling_schedule.get_seq_length() + 1,
            "Wrong sampling schedule length")
    
    def test_seq_dimensions(self):
        self.assertEqual(self.sampling_schedule.get_sampling_schedule().shape, (self.sampling_schedule.get_seq_length() + 1,2),
            "Wrong sampling schedule dimensions")
    
    def test_seq_values(self):
        self.assertEqual(self.sampling_schedule.get_sampling_schedule()[:,1].max(), self.sampling_schedule.get_sampling_schedule()[:,1].min(),
            "Number of shots not uniform")
        self.assertEqual(self.sampling_schedule.get_sampling_schedule()[:,1].max(), self.sampling_schedule.get_n_shots(),
            "Number of shots incorrect in schedule")
        self.assertEqual(self.sampling_schedule.get_sampling_schedule()[:,0].min(), 0,
            f"Minimum value of the linear sequence is not 0")
        self.assertEqual(self.sampling_schedule.get_sampling_schedule()[:,0].max(), self.max_m,
            f"Maximum value of the linear sequence is not {self.max_m}")
            