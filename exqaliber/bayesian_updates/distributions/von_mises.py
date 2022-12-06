"""Implementation of von Mises distribution."""
import warnings
from math import isclose, prod

import numpy as np
from scipy.special import ive as modified_bessel
from scipy.stats import vonmises

from .circular_distribution_base import (
    CIRCULAR_DISTRIBUTION,
    CircularDistributionBase,
)


class VonMises(CircularDistributionBase):
    r"""A von Mises continuous random variable.

    .. math::
        f(x, \mu, \kappa) = \frac{\exp(\kappa \cos(x - \mu))}
        {2 \pi I_0(\kappa)}

    For :math:`x \in [0, 2\pi) = S^1`, Location parameter - :math:`\mu
    \in S^1`,
    Concentration parameter - :math:`\kappa > 0`

    This distribution has the maximal entropy for a fixed location and
    scale. It is closed under pointwise multiplication of densities but
    not convolutions.

    Attributes
    ----------
    mu : float
        Location parameter in S^1
    kappa : float
        Concentration parameter > 0

    """

    def __init__(self, mu: float, kappa: float):
        r"""Initialise VonMises.

        Parameters
        ----------
        mu : float
            Mean value / location
        kappa : float
            Concentration parameter
        """
        if mu < -np.pi or mu >= np.pi:
            warnings.warn(
                f"Provided mu, {mu}, should be within the range [-pi, pi)"
            )
        self.mu = np.mod(mu, 2 * np.pi)
        if self.mu >= np.pi:
            self.mu = self.mu - 2 * np.pi

        if kappa <= 0:
            raise ValueError("Provided kappa must be strictly positive")
        self.kappa = kappa

        super().__init__(
            CIRCULAR_DISTRIBUTION.VON_MISES,
            {"mu": self.mu, "kappa": self.kappa},
        )

    def get_circular_moment(self, n: int = 1) -> complex:
        """Get the nth circular moment of the distribution.

        Parameters
        ----------
        n : int
            nth moment. n > 0

        Returns
        -------
        complex
            E[1i*nX]

        """
        max_n = int(abs(n))
        ratios = (
            VonMises.get_bessel_ratio(self.kappa, i) for i in range(max_n)
        )
        return prod(ratios) * np.exp(1j * n * self.mu)

    @staticmethod
    def generate_parameters_from_m1(m1: complex) -> tuple[float]:
        """Generate values for parameters mu, kappa from m1.

        Parameters
        ----------
        m1 : complex
            First circular moment of some distribution or sample

        Returns
        -------
        float :
            mu, location parameter in S^1
        float :
            kappa, concentration parameter > 0

        """
        return np.arctan2(
            m1.imag, m1.real
        ), VonMises.get_vm_concentration_param(m1)

    def sample(self, n: int = 10) -> np.ndarray:
        """Get n samples from the distribution.

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
        return vonmises.rvs(self.kappa, self.mu, size=n)

    @staticmethod
    def update(
        measurement: int, lamda: int, mu: float, kappa: float
    ) -> complex:
        """Get the first circular moment given a measurement.

        Updates the circular moment using a von Mises and Bernoulli
        likelihood.

        Parameters
        ----------
        measurement : int, {0,1}
            Bernoulli measurement outcome
        lamda : int
            Defines p(1) = 0.5*(1 - cos(lamda * mu))
        mu : float
            Location parameter of the current von Mises distribution
        kappa : float
            Scale parameter of the current von Mises distribution

        Returns
        -------
        complex
            First circular moment of the new distribution
        """
        sign = (-1) ** measurement
        bessel_vals = modified_bessel(
            [0, 1, lamda - 1, lamda, lamda + 1], float(kappa)
        )
        denom = bessel_vals[0] + sign * bessel_vals[3] * np.cos(lamda * mu)
        numer = np.exp(1j * mu) * (
            bessel_vals[1]
            + sign
            * 0.5
            * (
                np.exp(1j * lamda * mu) * bessel_vals[4]
                + np.exp(-1j * lamda * mu) * bessel_vals[2]
            )
        )
        posterior_moment = numer / denom

        return posterior_moment

    @staticmethod
    def expected_radius(grover_depth: int, mu: float, kappa: float) -> float:
        """Calculate the expected radius of the next posterior.

        Parameters
        ----------
        grover_depth : int
            Grover depth to estimate radius for
        mu : float
            Location parameter for the current von Mises distribution
        kappa : float
            Scale parameter for the current von Mises distribution
        t : int, optional
            Time step to estimate the radius for, by default current
            time

        Returns
        -------
        float
            Expected radius for the next step at the given Grover depth
            and von Mises distribution
        """
        lamda = 4 * grover_depth + 2
        return VonMises.expected_lambda_radius(lamda, mu, kappa)

    @staticmethod
    def expected_lambda_radius(lamda: int, mu: float, kappa: float) -> float:
        """Calculate the expected radius of the next step.

        See report for calculations.

        Parameters
        ----------
        lamda : int
            Defines the probability of observing a 1 as
            0.5(1-cos(lambda * theta))
        mu : float
            Location parameter for the current von Mises distribution
        kappa : float
            Scale parameter for the current von Mises distribution
        t : int, optional
            Time step to calculate the next radius for, by default use
            the current timestep

        Returns
        -------
        float
            Expected radius of the next step
        """
        kappa = float(kappa)
        lamda = np.array(lamda)

        constant_part = np.square(
            np.array([1, 0.5, 0.5])
            * modified_bessel(
                [np.ones(lamda.shape), lamda + 1, lamda - 1], kappa
            ).T
        ).sum(axis=1) + 0.5 * np.cos(2 * lamda * mu) * modified_bessel(
            [lamda + 1, lamda - 1], kappa
        ).prod(
            axis=0
        )
        phase_part = (
            modified_bessel(1, kappa)
            * modified_bessel([lamda + 1, lamda - 1], kappa).sum(axis=0)
            * np.cos(lamda * mu)
        )
        r_0 = constant_part + phase_part
        r_1 = constant_part - phase_part

        return 0.5 * (np.sqrt(r_0) + np.sqrt(r_1)) / modified_bessel(0, kappa)

    @staticmethod
    def eval_grover_radii(
        max_depth: int, mu: float, kappa: float
    ) -> np.ndarray:
        """Evaluate the expected radius up to a maximum Grover depth.

        Parameters
        ----------
        max_depth : int
            Maximum depth to evaluate Grover depth to
        mu : float
            Location parameter for the current von Mises distribution
        kappa : float
            Scale parameter for the current von Mises distribution

        Returns
        -------
        np.ndarray
            Expected radius for Grover depth values up to the given max
            depth
        """
        return np.ndarray(
            [
                VonMises.expected_radius(i_d, mu, kappa)
                for i_d in range(max_depth)
            ]
        )

    @staticmethod
    def eval_radii(max_depth: int, mu: float, kappa: float) -> np.ndarray:
        """Evaluate the expected radius up to a maximum lambda.

        Parameters
        ----------
        max_depth : int
            Maximum lambda depth to evaluate to
        mu : float
            Location parameter for the current von Mises distribution
        kappa : float
            Scale parameter for the current von Mises distribution

        Returns
        -------
        np.ndarray
            Expected radius for values of lambda up to the given max
            depth
        """
        return VonMises.expected_lambda_radius(
            np.array([i_d for i_d in range(max_depth)]), mu, kappa
        )

    @staticmethod
    def get_ber_prob(lamda: int, kappa: float, mu: float) -> float:
        """Get the marginal Bernoulli probability from a conditional VM.

        Assume that P(X = 1 | Theta = mu) = (1/2)* (1 - cos(lamda mu)).
        Then if Theta is distributed as VM(mu, kappa), then the
        returned probability is the marginal probability P(X = 1).

        Parameters
        ----------
        lamda : int
            Scale parameter for conditional Bernoulli distribution
        kappa : float
            Scale parameter of prior von-Mises distribution
        mu : float
            Location parameter of prior von-Mises distribution

        Returns
        -------
        float
            Marginal probability of observing a 1
        """
        return 0.5 * (
            1
            - np.cos(lamda * mu)
            * modified_bessel(lamda, kappa)
            / modified_bessel(0, kappa)
        )

    @staticmethod
    def get_bessel_ratio(x: float, v: int = 0, N: int = 10) -> float:
        """Compute I_{v+1}(x)/I_v(x).

        I_v is the vth modified Bessel function of the first kind
        Method as given by D. E. Amos in "Computation of Modified Bessel
        Functions and Their Ratios", and converted to pseudocode by Kurz
        et. al in "Recursive Nonlinear Filtering for Angular Data Based
        on Circular Distributions".

        Parameters
        ----------
        x : float
            Point at which to evaluate the ratio, x > 0
        v : int, optional
            Determines the orders of the Bessel functions in the ratio.
            Default value is 0, so gives a value for the magnitude of
            the first circular moment of the von Mises distribution with
            concentration parameter x.
        N : int, optional
            Number of discretisation steps

        Returns
        -------
        float
            Value of I_{v+1}(x)/I_v(x)

        """
        o = min(v, 10)
        r = np.zeros(N + 1)
        for i in range(N + 1):
            r[i] = x / (o + i + 0.5 + np.sqrt((o + i + 1.5) ** 2 + x**2))
        for i in range(1, N + 1):
            for k in range(N - i + 1):
                r[k] = x / (
                    o
                    + k
                    + 1
                    + np.sqrt((o + k + 1) ** 2 + (r[k + 1] / r[k]) * (x**2))
                )
        y = r[0]
        i = o
        while i > v:
            y = np.power((2 * i / x) + y, -1)
            i -= 1

        return y

    @staticmethod
    def get_vm_concentration_param(
        m1: complex, eps: float = pow(10, -9), v: int = 0, N: int = 10
    ) -> float:
        """Invert I_{v+1}(k)/I_v(k) = abs(m1) to precision eps.

        Parameters
        ----------
        m1 : complex
            First circular moment of von Mises distribution.
            m1 = exp(i mu) * I_{v+1}(k)/I_v(k).
        eps : float
            Precision to return k to.
        v : int, optional
            Inverts the ratio for higher order Bessel functions. Default
            is 0 for the first circular moment of von Mises
            distribution.
        N : int, optional
            Number of discretisations steps for calculating the Bessel
            ratio. Default is 10.

        Returns
        -------
        float
            For the interval [k_l, k_u] return (k_l + k_u)/2, where
            k_u - k_l < eps.

        """
        r = abs(m1)
        concentration_param_interval = np.array([0, 1], dtype=np.float128)
        found_upper_limit = False
        while not isclose(
            concentration_param_interval[0],
            concentration_param_interval[1],
            rel_tol=eps,
        ):
            k_trial = concentration_param_interval.mean()
            r_trial = VonMises.get_bessel_ratio(k_trial, v, N)
            # Increase the upper limit of the interval until we contain
            # k
            if not found_upper_limit:
                if r_trial > r:
                    found_upper_limit = True
                    concentration_param_interval[1] = k_trial
                else:
                    concentration_param_interval[
                        0
                    ] = concentration_param_interval[1]
                    concentration_param_interval[1] = (
                        2 * concentration_param_interval[1]
                    )
            else:
                if r_trial > r:
                    concentration_param_interval[1] = k_trial
                else:
                    concentration_param_interval[0] = k_trial
        return concentration_param_interval.mean()
