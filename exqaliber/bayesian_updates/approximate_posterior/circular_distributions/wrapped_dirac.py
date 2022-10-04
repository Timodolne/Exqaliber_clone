import numpy as np

from .circular_distribution_base import (CIRCULAR_DISTRIBUTION,
                                         CircularDistributionBase)


class WrappedDirac(CircularDistributionBase):
    '''Wrapped Dirac distribution

    Defined by f(x;gamma,beta) = sum_{j=1}^L gamma_j delta(x - beta_j)
    Dirac delta function, delta(.)
    Dirac positions, beta_1, ..., beta_L in S^1
    Weights, gamma_1, ..., gamma_L > 0 with sum_{j=1}^L gamma_j = 1

    This distribution is just the most general discrete probability
    distribution on S^1

    Attributes
    ----------
    beta : np.ndarray(L, dtype=np.float32)
        Dirac positions
    gamma: np.ndarray(L, dtype-np.float32)
        Weights

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
        # beta_j should take values in [0, 2*pi)
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
            E[1i*nX]

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
            Returns n samples from the distribution. Takes values in
            [0, 2*pi)

        '''
        samples = np.random.uniform(0,1,n)


        raise NotImplementedError

    @staticmethod
    def generate_parameters_from_m1(m1: complex) -> tuple[np.ndarray]:
        '''Generate values for distribution parameters beta, gamma matching first circular moment m1

        As given in "Recursive Nonlinear Filtering for Angular Data
        Based on Circular Distributions" Kurz et al.

        Parameters
        ----------
        m1 : complex
            First circular moment for a distribution

        Returns
        -------
        np.ndarray(3) :
            Beta values for wrapped dirac distribution with matched m1
        np.ndarray(3)
            Gamma values for wrapped dirac distribution with matched m1

        '''
        mu = np.arctan2(m1.imag, m1.real)
        alpha = np.arccos(1.5*abs(m1) - 0.5)
        beta = np.mod(np.array([mu - alpha, mu, mu +alpha]), 2*np.pi)
        gamma = (1/3)*np.ones(3)
        return beta, gamma

    @staticmethod
    def generate_parameters_from_m1_m2(m1: complex, m2: complex, a : float = 0.5) -> tuple[np.ndarray]:
        '''Generate values for distribution parameters beta, gamma matching first and second circular moments m1, m2

        As given in "Deterministic Approximation of Circular Densities
        with Symmetric Dirac Mixtures Based on "Two Circular Moments"
        Kurz et al.

        Parameters
        ----------
        m1 : complex
            First circular moment for a distribution
        m2 : complex
            Second circular moment for a distribution
        a : float, optional
            Parameter in [0,1], by default 0.5

        Returns
        -------
        np.ndarray(5) :
            Beta values for wrapped dirac distribution with matched m1,
            m2
        np.ndarray(5)
            Gamma values for wrapped dirac distribution with matched m1,
            m2

        '''
        # Extract mu
        mu = np.arctan2(m1.imag, m1.real)
        m_1 = abs(m1)
        m_2 = abs(m2)

        # Obtain weights
        gamma_5_min = (4*(m_1**2) - 4*m_1 - m2 + 1)/(4*m_1 - m_2 - 3)
        gamma_5_max = (2*(m_1**2) - m_2 - 1)/(4*m_1 - m_2 - 3)
        gamma_5 = gamma_5_min + a*(gamma_5_max - gamma_5_min)
        gamma = ((1 - gamma_5)/4)* np.ones(5)
        gamma[5] = gamma_5

        # Obtain dirac positions
        c_1 = (2/(1 - gamma_5))*(m_1-gamma_5)
        c_2 = (1/(1 - gamma_5))*(m_2 - gamma_5) + 1
        x_2 = (2*c_1 + np.sqrt(4*(c_1**2) - 8*((c_1**2) - c_2)))/4
        x_1 = c_1 - x_2
        phi_1 = np.arccos(x_1)
        phi_2 = np.arccos(x_2)
        beta = np.mod(mu*np.ones(5) + np.array([-phi_1, phi_1, -phi_2, phi_2, 0]), 2*np.pi)
        return beta, gamma
