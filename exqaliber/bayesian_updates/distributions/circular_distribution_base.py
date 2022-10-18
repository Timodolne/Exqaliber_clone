"""Base class and Enum type for circular distributions."""
from abc import ABC, abstractmethod
from enum import Enum

import numpy as np


class CIRCULAR_DISTRIBUTION(Enum):
    r"""Type of circular distribution as an Enum."""

    WRAPPED_NORMAL = 0
    VON_MISES = 1
    WRAPPED_DIRAC = 2
    WRAPPED_DIRICHLET = 3


class CircularDistributionBase(ABC):
    r"""Abstract base class for circular distributions.

    Attributes
    ----------
    __type : CIRCULAR_DISTRIBUTION
        Type of the distribution
    __parameters: dict[str, float | np.ndarray]
        Parameters that uniquely define the distribution

    """

    def __init__(
        self,
        distribution: CIRCULAR_DISTRIBUTION,
        parameters: dict[str, float | np.ndarray],
    ):
        r"""Initialise CircularDistributionBase.

        Parameters
        ----------
        distribution : CIRCULAR_DISTRIBUTION
            Type of the distribution
        parameters: dict[str, float | np.ndarray]
            Parameters that uniquely define the distribution

        """
        self.__type = distribution
        self.__parameters = parameters

    @abstractmethod
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
            [0, 2*pi)

        """
        pass

    @abstractmethod
    def get_circular_moment(self, n: int = 1) -> complex:
        r"""Get the nth circular moment of the distribution.

        Parameters
        ----------
        n : int
            nth moment. n > 0

        Returns
        -------
        complex
            E[1j*nX]

        """
        pass

    def get_circular_variance(self) -> float:
        r"""Get the circular variance of the distribution.

        Returns
        -------
        float
            Circular variance = 1 - abs(m1), for m1 the circular mean

        """
        return 1 - abs(self.get_circular_moment())

    def get_circular_standard_deviation(self) -> float:
        r"""Get the circular standard deviation of the distribution.

        Returns
        -------
        float
            Circular standard variance sqrt{log(1 / abs(m1)^2)}, for m1
            the circular mean

        """
        return np.sqrt(-2 * np.log(abs(self.get_circular_moment())))

    def get_circular_dispersion(self) -> float:
        r"""Get the circular dispersion of the distribution.

        Returns
        -------
        float
            Circular dispersion delta = (1 - abs(m2))/(2*abs(m2)^2)

        """
        return (1 - self.get_circular_moment(2)) / (
            2 * self.get_circular_moment() ** 2
        )

    def get_type(self) -> CIRCULAR_DISTRIBUTION:
        r"""Get the type of the distribution.

        Returns
        -------
        CIRCULAR_DISTRIBUTION
            The type of the distribution

        """
        return self.__type

    def get_parameters(self) -> dict[str, float | np.ndarray]:
        r"""Get the parameters that uniquely define the distribution.

        Returns
        -------
        dict[str, float | np.ndarray]
            Parameters that uniquely define the distribution e.g.
            WN(mu, sigma), VM(mu,kappa), WD(gamma,beta)

        """
        return self.__parameters
