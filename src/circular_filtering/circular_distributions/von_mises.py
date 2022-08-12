import numpy as np
from .circular_distribution_base import CircularDistributionBase, CIRCULAR_DISTRIBUTION


class VonMises(CircularDistributionBase):
    '''von Mises distribution 

    Defined by f(x, \mu, \kappa) = \frac{1}{2 \pi I_0(\kappa)} \exp(\kappa \cos(x - \mu))

    For x \in [0, 2 \pi) = S^1, 
    Location parameter, \mu \in S^1, 
    Concentration parameter, \kappa > 0
    
    This distribution is obtained by wrapping a 1-d Gaussian around the unit circle and adding up all probability mass that is 
    wrapped to the same point

    Attributes
    ----------
    mu : float
        Location parameter in S^1
    kappa : float
        Concentration parameter > 0

    Inherited Attributes
    --------------------
    distribution : CIRCULAR_DISTRIBUTION.VON_MISES
    parameters : dict[str, float]
        mu : self.mu
        kappa : self.kappa

    Methods
    -------
    generate_parameters_from_m1(m1)
        Generate values for distribution parameters mu, kappa from m1
    get_bessel_ratio(x, v, N)
        Compute I_{v+1}(x)/I_v(x)
    get_vm_concentration_param(m1, eps, v, N)
        Invert I_{v+1}(k)/I_v(k) = |m1| to precision \pm \varepsilon

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

    def __init__(self, mu : float, kappa: float):
        self.mu = np.mod(mu, 2 * np.pi)
        self.kappa = abs(kappa)

        super.__init__(CIRCULAR_DISTRIBUTION.VON_MISES,
            {
                'mu': self.mu,
                'kappa': self.kappa
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
    
    @staticmethod
    def get_bessel_ratio(x : float, v : int = 0, N : int = 10 ) -> float:
        '''Compute I_{v+1}(x)/I_v(x)
        I_v is the vth modified Bessel function of the first kind

        Method as given by D. E. Amos in "Computation of Modified Bessel Functions and Their Ratios",
        and converted to pseudocode by Kurz et. al in "Recursive Nonlinear Filtering for Angular Data
        Based on Circular Distributions".
        
        Parameters
        ----------
        x : float
            Point at which to evaluate the ratio. x > 0
        v : int, optional
            Determines the orders of the Bessel functions in the ratio. Default value
            is 0, so gives a value for the magnitude of the first circular moment of 
            the von Mises distribution with concentration parameter x.
        N : int, optional
            Number of discretisation steps

        Returns
        -------
        float : 
            Value of I_{v+1}(x)/I_v(x)
        '''

        o = min(v, 10)
        r = np.zeros(N +1)
        for i in range(N + 1):
            r[i] = x/(o + i + 0.5 + np.sqrt((o + i + 1.5)**2 + x**2))
        for i in range(1,N+1):
            for k in range(N-i+1):
                r[k] = x/(o + k + 1 + np.sqrt((o + k + 1)**2 + (r[k+1]/r[k])*(x**2)))
        y = r[0]
        i = o
        while i > v:
            y = np.power((2*i/x) + y ,-1)
            i -= 1

        return y

    @staticmethod
    def get_vm_concentration_param(m1: complex, eps: float = pow(2,-15), v : int = 0, N : int = 10) -> float:
        '''Invert I_{v+1}(k)/I_v(k) = |m1| to precision \pm \varepsilon

        Parameters
        ----------
        m1 : complex
            First circular moment of von Mises distribution. 
            m1 = exp(i mu) * I_{v+1}(k)/I_v(k).
        eps : float
            Precision to return k to. 
        v : int, optional
            Inverts the ratio for higher order Bessel functions. Default is 0 for the first
            circular moment of von Mises distribution.
        N : int, optional
            Number of discretisations steps for calculating the Bessel ratio. Default is 10.

        Returns
        -------
        float : 
            For the interval [k_l, k_u] return (k_l + k_u)/2, where k_u - k_l < eps.
        '''
        
        r = abs(m1)
        concentration_param_interval = np.array([0,1],dtype=np.float32)
        found_upper_limit = False

        while concentration_param_interval[1] - concentration_param_interval[0] > eps:
            k_trial = concentration_param_interval.mean()
            r_trial = VonMises.get_bessel_ratio(k_trial, v, N)

            # Increase the upper limit of the interval until we contain k
            if not found_upper_limit:
                if r_trial > r:
                    found_upper_limit = True
                    concentration_param_interval[1] = k_trial
                else:
                    concentration_param_interval[0] = concentration_param_interval[1]
                    concentration_param_interval[1] = 2*concentration_param_interval[1]
            else:
                if r_trial > r:
                    concentration_param_interval[1] = k_trial
                else:
                    concentration_param_interval[0] = k_trial

        return concentration_param_interval.mean()
