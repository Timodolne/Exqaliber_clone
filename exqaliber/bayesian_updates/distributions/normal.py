"""Normal distribution for Bayesian Updates."""
from typing import Tuple

import numpy as np
from scipy.stats import norm


class Normal:
    """Normal distribution."""

    def __init__(self, mu: float, sigma: float) -> None:

        if sigma <= 0:
            raise ValueError("Standard deviation must be positive")

        self.__mu = mu
        self.__sigma = sigma

        self.__parameters = {"mu": mu, "sigma": sigma}

    """
    Parameters / Getters for the base distribution
    """

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

    """
    Update method given a Bernoulli measurement.
    """

    @staticmethod
    def get_expected_bias(lamda: int, mu: float, sigma: float) -> float:
        """Get the expected bias the updated normal distribution.

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
            Expected bias
        """
        exp = np.exp(-0.5 * lamda**2 * sigma**2)
        trig = np.cos(lamda * mu)
        return exp * trig

    @staticmethod
    def get_chi(lamda: int, mu: float, sigma: float) -> float:
        """Get the expected bias the updated normal distribution.

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
            Expected bias
        """
        exp = (-1) * lamda * np.exp(-0.5 * lamda**2 * sigma**2)
        trig = np.sin(lamda * mu)
        return exp * trig

    @staticmethod
    def get_first_moment_posterior(
        measurement: int, lamda: int, mu: float, sigma: float
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

        Returns
        -------
        float
            First moment of the posterior
        """
        sign = (-1) ** measurement
        b = Normal.get_expected_bias(lamda, mu, sigma)
        chi = Normal.get_chi(lamda, mu, sigma)

        denom = 1 + sign * b
        numer = mu + sign * (sigma**2 * chi + mu * b)

        return numer / denom

    @staticmethod
    def get_second_moment_posterior(
        measurement: int, lamda: int, mu: float, sigma: float
    ):
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

        Returns
        -------
        float
            Second moment of the posterior
        """
        sign = (-1) ** measurement
        b = Normal.get_expected_bias(lamda, mu, sigma)
        chi = Normal.get_chi(lamda, mu, sigma)

        denom = 1 + sign * b
        numer = (
            sigma**2
            + mu**2
            + sign
            * (
                (sigma**2 + mu**2 - lamda**2 * sigma**4) * b
                + 2 * mu * chi * sigma**2
            )
        )

        return numer / denom

    @staticmethod
    def update(
        measurement: int, lamda: int, mu: float, sigma: float
    ) -> Tuple[float, float]:
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

        Returns
        -------
        Tuple[float, float]
            Mean and standard deviation of the new distribution
        """
        posterior_mu = Normal.get_first_moment_posterior(
            measurement, lamda, mu, sigma
        )
        second_moment = Normal.get_second_moment_posterior(
            measurement, lamda, mu, sigma
        )
        posterior_var = second_moment - posterior_mu**2
        posterior_sigma = np.sqrt(posterior_var)

        return posterior_mu, posterior_sigma

    @staticmethod
    def get_variance_reduction_factor(
        lamda: int, mu: float, sigma: float
    ) -> float:
        """Get the variance reduction factor for given lambda.

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
            Variance reduction factor
        """
        b = Normal.get_expected_bias(lamda, mu, sigma)
        chi = Normal.get_chi(lamda, mu, sigma)

        if np.abs(b - 1) < 1e-8:
            return 0
        else:
            return chi**2 / (1 - b**2)

    @staticmethod
    def eval_lambdas(
        lambdas: np.ndarray, mu: float, sigma: float
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

        Returns
        -------
        np.ndarray
            Variance reduction factor for given lambdas
        """
        return np.array(
            [
                Normal.get_variance_reduction_factor(lamda, mu, sigma)
                for lamda in lambdas
            ]
        )

    def confidence_interval(self, alpha):
        """Return (1-alpha)% confidence interval."""
        theta_min = norm.ppf(alpha / 2, self.mean, self.standard_deviation)
        theta_max = norm.ppf(1 - alpha / 2, self.mean, self.standard_deviation)

        return theta_min, theta_max
