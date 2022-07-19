from abc import ABC
from enum import Enum

class NOISE_MODEL(Enum):
    NOISELESS = 0
    DEPOLARISING = 1

class BaseNoiseModel(ABC):
    '''Abstract base class for noise models
    
    Attributes
    ----------
    __type : NOISE_MODEL
        Noise model
    
    Methods
    -------
    get_type()
        Get the noise model
    '''

    def __init__(self, noise_model: NOISE_MODEL):
        '''
        Parameters
        ----------
        noise_model : NOISE_MODEL
            Noise model being used
        '''

        self.__type = noise_model
    
    def get_type(self) -> NOISE_MODEL:
        '''Get the noise model
        
        Returns
        -------
        NOISE_MODEL :
            Noise model being used
        '''

        return self.__type
