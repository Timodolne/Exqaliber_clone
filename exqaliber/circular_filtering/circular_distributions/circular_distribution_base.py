from abc import ABC, abstractmethod
from enum import Enum

import numpy as np


class CIRCULAR_DISTRIBUTION(Enum):
    WRAPPED_NORMAL = 0
    VON_MISES = 1
    WRAPPED_DIRAC = 2
    WRAPPED_DIRICHLET = 3
    

class CircularDistributionBase(ABC):
    '''Abstract base class for circular distributions

    Attributes
    ----------
    __type : CIRCULAR_DISTRIBUTION
        Type of the distribution
    __parameters: dict[str, float | np.ndarray]
        Parameters that uniquely define the distribution

    Methods
    -------
    get_circular_variance()
        Get the circular variance of the distribution
    get_circular_standard_deviation()
        Get the circular variance of the distribution
    get_circular_dispersion()
        Gets the circular dispersion of the distribution
    get_type()
        Gets the type of the distribution
    get_parameters()
        Gets the parameters that uniquely define the distribution

    Abstract Methods
    ----------------
    sample(n)
        Get n samples from the distribution
    get_circular_moment(n)
        Get the nth circular moment of the distribution
    '''

    def __init__(self, distribution : CIRCULAR_DISTRIBUTION, parameters : dict[str, float | np.ndarray]):
        '''
        Parameters
        ----------
        __type : CIRCULAR_DISTRIBUTION
            Type of the distribution
        __parameters: dict[str, float | np.ndarray]
            Parameters that uniquely define the distribution
        '''

        self.__type = distribution
        self.__parameters = parameters
        
    @abstractmethod
    def sample(self, n : int = 10) -> np.ndarray:
        '''Get n samples from the distribution
        
        Parameters
        ----------
        n : int, optional
            Number of samples. Default is 10

        Returns
        -------
        np.ndarray(n, np.float)
            Returns n samples from the distribution. Takes values in [0, 2\pi)
        '''

        pass

    @abstractmethod
    def get_circular_moment(self, n : int = 1) -> complex:
        '''Get the nth circular moment of the distribution
        
        Parameters
        ----------
        n : int
            nth moment. n > 0
        
        Returns
        -------
        complex : 
            \mathbb{E}[inX]
        '''

        pass


    def get_circular_variance(self) -> float:
        '''Get the circular variance of the distribution
        
        Returns
        -------
        float : 
            Circular variance = 1 - |m_1|, for m_1 the circular mean
        '''

        return 1 - abs(self.get_circular_moment())
    
    def get_circular_standard_deviation(self) -> float:
        '''Gets the circular standard deviation of the distribution

        Returns
        -------
        float :
            Circular standard variance \sqrt{\log(1 / |m_1|^2)}, for m_1 the circular mean        
        '''

        return np.sqrt(-2*np.log(abs(self.get_circular_moment())))
    
    def get_circular_dispersion(self) -> float:
        '''Gets the circular dispersion of the distribution

        Returns
        -------
        float : 
            Circular disperson \delta = \frac{1 - |m_2|}{2|m_2|^2}
        '''

        return (1 - self.get_circular_moment(2))/(2* self.get_circular_moment()**2)

    def get_type(self) -> CIRCULAR_DISTRIBUTION:
        '''Gets the type of the distribution

        Returns
        -------
        CIRCULAR_DISTRIBUTION : 
            The type of the distribution
        '''

        return self.__type
    
    def get_parameters(self) -> dict[str, float | np.ndarray]:
        '''Get the parameters that uniquely define the distribution
        
        Returns
        -------
        dict[str, float | np.ndarray] : 
            Parameters that uniquely define the distribution e.g. 
            WN(\mu, \sigma), VM(\mu, \kappa), WD(\boldsymbol\gamma, \boldsymbol\beta)
        '''

        return self.__parameters
