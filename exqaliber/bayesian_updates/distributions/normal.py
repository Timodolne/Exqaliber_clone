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
        return np.random.normal(self.mu, self.variance, size=n)

    """
    Update method given a Bernoulli measurement.
    """

    @staticmethod
    def update(
        measurement: int, lamda: int, mu: float, kappa: float
    ) -> Tuple[float, float]:
        """Get the mean and variance of the updated normal distribution.

        Updates the the mean and variance using a normal prior and
        Bernoulli likelihood.

        Parameters
        ----------
        measurement : int, {0,1}
            Bernoulli measurement outcome
        lamda : int
            Defines p(1) = 0.5*(1 - cos(lamda * mu))
        mu : float
            Location parameter of the current normal distribution
        kappa : float
            Scale parameter of the current normal distribution

        Returns
        -------
        Tuple[float, float]
            Mean and variance of the new distribution
        """
        raise NotImplementedError()
