import numpy as np
from .circular_distribution_base import CircularDistributionBase, CIRCULAR_DISTRIBUTION


class WrappedNormal(CircularDistributionBase):
    '''Wrapped Normal distribution 

    Defined by f(x, \mu, \sigma) = \frac{1}{2 \pi \sigma^2} \sum_{k = -\infty}^\infty \exp(- \frac{(x - \mu + 2 \pi k)^2}{2 \sigma^2})
    
    For
    x \in [0, 2 \pi) = S^1, 
    Location parameter, \mu \in S^1, 
    Concentration parameter, \sigma > 0
    
    This distribution is obtained by wrapping a 1-d Gaussian around the unit circle and adding up all probability mass that is 
    wrapped to the same point. It is closed under convolutions but not under pointwise multiplication. 

    Attributes
    ----------
    mu : float
        Location parameter in S^1
    sigma : float
        Concentration parameter > 0

    Inherited Attributes
    --------------------
    distribution : CIRCULAR_DISTRIBUTION.WRAPPED_NORMAL
    parameters : dict[str, float]
        mu : self.mu
        sigma : self.sigma

    Methods
    -------
    generate_parameters_from_m1(m1)
        Generate values for distribution parameters mu, sigma from m1

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

    def __init__(self, mu : float, sigma: float):
        self.mu = np.mod(mu, 2 * np.pi)
        self.sigma = abs(sigma)

        super.__init__(CIRCULAR_DISTRIBUTION.WRAPPED_NORMAL,
            {
                'mu': self.mu,
                'sigma': self.sigma
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

        return np.exp(1j *n * self.mu  - ((n * self.sigma)**2 / 2))

    @staticmethod
    def generate_parameters_from_m1(m1: complex) -> tuple[float]:
        '''Generate values for distribution parameters mu, sigma from m1
        
        Parameters
        ----------
        m_1 : complex
            First circular moment of some distribution or sample

        Returns
        -------
        float : 
            \mu, location parameter in S^1
        float : 
            \sigma, concentration parameter > 0

        '''
        
        return np.arctan2(m1.imag, m1.real), np.sqrt(-2*np.log(abs(m1)))
    
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

        raise NotImplementedError