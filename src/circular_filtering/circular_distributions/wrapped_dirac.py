import numpy as np
from .circular_distribution_base import CircularDistributionBase, CIRCULAR_DISTRIBUTION


class WrappedDirac(CircularDistributionBase):
    '''Wrapped Dirac distribution 

    Defined by f(x;\boldsymbol\gamma,\boldsymbol\beta) = \sum_{j=1}^L \gamma_j \delta(x - \beta_j)

    Dirac delta function, \delta(\cdot)  
    Dirac positions, \beta_1, \ldots, \beta_L \in S^1
    Weights, \gamma_1, \ldots, \gamma_L > 0 with \sum_{j=1}^L \gamma_j = 1

    This distribution is just the most general discrete probability distribution on S^1

    Attributes
    ----------
    beta : np.ndarray(L, dtype=np.float32)
        Dirac positions
    gamma: np.ndarray(L, dtype-np.float32)
        Weights

    Inherited Attributes
    --------------------
    distribution : CIRCULAR_DISTRIBUTION.WRAPPED_DIRAC
    parameters : dict[str, np.ndarray(L, dtype=np.float32)]
        beta : self.beta
        gamma : self.gamma

    Methods
    -------

    
    Inherited Methods
    -----------------
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
    '''

    def __init__(self, gamma : np.ndarray, beta: np.ndarray):
        '''
        Parameters
        ----------
        gamma: np.ndarray(L, dtype-np.float32)
            Weights
        beta : np.ndarray(L, dtype=np.float32)
            Dirac positions  
        '''

        # \beta_j should take values in [0, 2\pi)
        self.beta = np.mod(beta, 2* np.pi)
        # Gamma should sum to 1
        self.gamma = gamma/gamma.sum()
        assert beta.shape == gamma.shape, "Beta and Gamma should have the same length"

        super.__init__(CIRCULAR_DISTRIBUTION.WRAPPED_DIRAC,
            {
                'beta': self.beta,
                'gamma': self.gamma
            }
        )
    
    def get_circular_moment(self, n: int = 1) -> complex:
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

        return np.dot(self.gamma, np.exp(1j*n *self.beta))
    
    def sample(self, n: int = 10) -> np.ndarray:
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

        samples = np.random.uniform(0,1,n)


        raise NotImplementedError