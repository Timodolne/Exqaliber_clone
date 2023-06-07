"""Normal distribution for Bayesian Updates."""
import numba
import numpy as np
from scipy.stats import norm


@numba.njit()
def get_expected_bias(lamda: int, mu: float, sigma: float) -> float:
    """Evaluate the expected bias of a normal distribution.

    This bias is computed with respect to a given normal distribution
    and Grover circuit depth, d, for 2(2d+1) = lambda.

    Parameters
    ----------
    lamda : int
        Defines p(1) = 0.5*(1 - cos(lamda * mu))
    mu : float
        Location parameter of the current normal distribution
    sigma : float
        Scale parameter of the current normal distribution

    Returns
    -------
    float
        Expected bias at lambda, mu, sigma
    """
    exp = np.exp(-0.5 * lamda**2 * sigma**2)
    trig = np.cos(lamda * mu)
    return exp * trig


@numba.njit()
def get_chi(lamda: int, mu: float, sigma: float) -> float:
    """Evaluate the chi function.

    Evaluate the chi function with respect to a given normal
    distribution and Grover circuit depth, d, for 2(2d+1) = lambda.

    Parameters
    ----------
    lamda : int
        Defines p(1) = 0.5*(1 - cos(lamda * mu))
    mu : float
        Location parameter of the current normal distribution
    sigma : float
        Scale parameter of the current normal distribution

    Returns
    -------
    float
        Chi function at lambda, mu, sigma
    """
    exp = (-1) * lamda * np.exp(-0.5 * lamda**2 * sigma**2)
    trig = np.sin(lamda * mu)
    return exp * trig


@numba.njit()
def get_first_moment_posterior(
    measurement: int,
    lamda: int,
    mu: float,
    sigma: float,
    zeta: float = 0,
) -> float:
    """Get 1st moment of posterior normal dist, given measurement.

    Calculates the first moment using a normal prior and
    Bernoulli likelihood.

    Parameters
    ----------
    measurement : int, {0,1}
        Bernoulli measurement outcome
    lamda : int
        Defines p(1) = 0.5*(1 - cos(lamda * mu))
    mu : float
        Location parameter of the current normal distribution
    sigma : float
        Scale parameter of the current normal distribution
    zeta : float
        Depolarising noise parameter

    Returns
    -------
    float
        First moment of the posterior
    """
    sign = (-1) ** measurement
    b = get_expected_bias(lamda, mu, sigma)
    chi = get_chi(lamda, mu, sigma)
    noise = np.exp(-lamda * zeta)

    denom = 1 + sign * b * noise
    numer = mu + sign * (sigma**2 * chi + mu * b) * noise

    return numer / denom


@numba.njit()
def get_second_moment_posterior(
    measurement: int, lamda: int, mu: float, sigma: float, zeta: float = 0
) -> float:
    """Get 2nd moment of posterior normal dist, given measurement.

    Calculates the second moment using a normal prior and
    Bernoulli likelihood.

    Parameters
    ----------
    measurement : int, {0,1}
        Bernoulli measurement outcome
    lamda : int
        Defines p(1) = 0.5*(1 - cos(lamda * mu))
    mu : float
        Location parameter of the current normal distribution
    sigma : float
        Scale parameter of the current normal distribution
    zeta : float
        Depolarising noise parameter

    Returns
    -------
    float
        Second moment of the posterior
    """
    sign = (-1) ** measurement
    b = get_expected_bias(lamda, mu, sigma)
    chi = get_chi(lamda, mu, sigma)
    noise = np.exp(-lamda * zeta)

    denom = 1 + sign * b * noise
    numer = (
        sigma**2
        + mu**2
        + sign
        * (
            (sigma**2 + mu**2 - lamda**2 * sigma**4) * b
            + 2 * mu * chi * sigma**2
        )
        * noise
    )

    return numer / denom


@numba.njit()
def get_variance_reduction_factor(
    lamda: int, mu: float, sigma: float, zeta: float = 0
) -> float:
    """Get the variance reduction factor for given lambda.

    Note that if |b| = 1, the variance reduction factor is 0.

    Parameters
    ----------
    lamda : int
        Defines p(1) = 0.5*(1 - cos(lamda * mu))
    mu : float
        Location parameter of the current normal distribution
    sigma : float
        Scale parameter of the current normal distribution
    zeta:
        Depolarising noise parameter

    Returns
    -------
    float
        Variance reduction factor
    """
    b = get_expected_bias(lamda, mu, sigma)
    chi = get_chi(lamda, mu, sigma)
    noise = np.exp(-lamda * zeta)

    if np.abs(b - 1) < 1e-8:
        return 0
    else:
        return (noise * chi**2) / (1 - noise * b**2)


class Normal:
    """Normal distribution."""

    def __init__(self, mu: float, sigma: float) -> None:
        if sigma <= 0:
            raise ValueError("Standard deviation must be positive")

        self.__mu = mu
        self.__sigma = sigma

        self.__parameters = {"mu": mu, "sigma": sigma}

    @property
    def mean(self) -> float:
        """Get the mean of the normal distribution."""
        return self.__mu

    @property
    def standard_deviation(self) -> float:
        """Get the standard deviation of the normal distribution."""
        return self.__sigma

    @property
    def variance(self) -> float:
        """Get the variance of the normal distribution."""
        return self.standard_deviation**2

    def get_parameters(self) -> dict[str, float | np.ndarray]:
        r"""Get the parameters that uniquely define the distribution.

        Returns
        -------
        dict[str, float | np.ndarray]
            Parameters that uniquely define the distribution e.g.
            WN(mu, sigma), VM(mu,kappa), WD(gamma,beta)

        """
        return self.__parameters

    def sample(self, n: int = 10) -> np.ndarray:
        r"""Get n samples from the distribution.

        Parameters
        ----------
        n : int, optional
            Number of samples. Default is 10

        Returns
        -------
        np.ndarray(n, np.float)
            Returns n samples from the distribution. Takes values in
            (-\inf, +\inf)

        """
        return np.random.normal(self.mean, self.standard_deviation, size=n)

    @staticmethod
    def update(
        measurement: int, lamda: int, mu: float, sigma: float, zeta: float = 0
    ) -> tuple[float, float]:
        """Get the mean and std of the updated normal distribution.

        Updates the mean and standard deviation using a normal prior and
        Bernoulli likelihood.

        Parameters
        ----------
        measurement : int, {0,1}
            Bernoulli measurement outcome
        lamda : int
            Defines p(1) = 0.5*(1 - cos(lamda * mu))
        mu : float
            Location parameter of the current normal distribution
        sigma : float
            Scale parameter of the current normal distribution
        zeta: float
            Depolarising noise parameter

        Returns
        -------
        tuple[float, float]
            Mean and standard deviation of the new distribution
        """
        posterior_mu = get_first_moment_posterior(
            measurement, lamda, mu, sigma, zeta
        )
        second_moment = get_second_moment_posterior(
            measurement, lamda, mu, sigma, zeta
        )
        posterior_var = second_moment - posterior_mu**2
        posterior_sigma = np.sqrt(posterior_var)

        return posterior_mu, posterior_sigma

    @staticmethod
    @numba.jit()
    def eval_lambdas(
        lambdas: np.ndarray, mu: float, sigma: float, zeta: float = 0
    ) -> np.ndarray:
        """Get the variance reduction factor for lambdas.

        Parameters
        ----------
        lambdas : np.ndarray
            Defines p(1) = 0.5*(1 - cos(lamda * mu))
        mu : float
            Location parameter of the current normal distribution
        sigma : float
            Scale parameter of the current normal distribution
        zeta : float
            Depolarising noise parameter

        Returns
        -------
        np.ndarray
            Variance reduction factor for given lambdas
        """
        return np.array(
            [
                get_variance_reduction_factor(lamda, mu, sigma, zeta)
                for lamda in lambdas
            ]
        )

    def confidence_interval(self, alpha: float) -> tuple[float]:
        """Return (1-alpha)% confidence interval.

        Confidence interval is generated from the current Normal
        distribution.

        Parameters
        ----------
        alpha : float
            Confidence level for interval.

        Returns
        -------
        tuple[float]
            Lower and upper bounds for the confidence interval.
        """
        theta_min = norm.ppf(alpha / 2, self.mean, self.standard_deviation)
        theta_max = norm.ppf(1 - alpha / 2, self.mean, self.standard_deviation)

        return theta_min, theta_max
