"""Normal distribution for Bayesian Updates."""
from typing import Tuple

import numpy as np


class Normal:
    """Normal distribution."""

    def __init__(self, mu: float, var: float) -> None:

        if var <= 0:
            raise ValueError("Variance must be positive")

        self.__mu = mu
        self.__var = var

        self.__parameters = {"mu": mu, "variance": var}

    """
    Parameters / Getters for the base distribution
    """

    @property
    def mean(self) -> float:
        """Get the mean of the normal distribution."""
        return self.__mu

    @property
    def variance(self) -> float:
        """Get the variance of the normal distribution."""
        return self.__var

    @property
    def standard_deviation(self) -> float:
        """Get the standard deviation of the normal distribution."""
        return np.sqrt(self.variance)

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
        return np.random.normal(self.__mu, self.__var, size=n)

    """
    Update method given a Bernoulli measurement.
    """

    @staticmethod
    def get_expected_bias(self, lamda: int, mu: float, sigma: float) -> float:
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
        return np.exp(-0.5 * lamda**2 * sigma**2) * np.cos(lamda * mu)

    @staticmethod
    def get_chi(self, lamda: int, mu: float, sigma: float) -> float:
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
        return (
            (-1)
            * lamda
            * np.exp(-0.5 * lamda**2 * sigma**2)
            * np.sin(lamda * mu)
        )

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
            + mu
            + sign
            * sigma**2
            * (
                (1 + mu**2 / sigma**2 - lamda**2 * sigma**2) * b
                - 2 * mu * chi
            )
        )

        return numer / denom

    @staticmethod
    def update(
        measurement: int, lamda: int, mu: float, sigma: float
    ) -> Tuple[float, float]:
        """Get the mean and variance of the updated normal distribution.

        Updates the mean and variance using a normal prior and
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
            Mean and variance of the new distribution
        """
        posterior_mu = Normal.get_first_moment_posterior(
            measurement, lamda, mu, sigma
        )
        second_moment = Normal.get_second_moment_posterior(
            measurement, lamda, mu, sigma
        )
        posterior_var = second_moment - posterior_mu**2

        return posterior_mu, posterior_var
