"""Bayesian update model."""
from typing import Tuple

import numpy as np
from scipy.special import ive as modified_bessel

from exqaliber.sampling_schedule.fixed_sampling_schedule.base import (
    BaseSamplingSchedule,
)

from .distributions.von_mises import VonMises


class BayesianModel:
    """Bayesian model for sequential estimation.

    Attributes
    ----------
    t : int
        Number of measurements taken so far
    prior: VonMises
        Prior distribution for theta
    estimated_params: List[Tuple[float,float]]
        Estimated parameters for (mu, kappa) at each time step
    measurements: List[int]
        Measurement at time t
    grover_depths: List[int]
        Grover depth at time t
    """

    def __init__(self, prior: VonMises) -> None:
        """Initialise BayesianModel.

        Parameters
        ----------
        prior : VonMises
            Prior distribution for the Bayesian updates
        """
        self.t = 0
        self.prior = prior

        self.estimated_params = [tuple(prior.get_parameters().values())]
        self.posterior_moments = [prior.get_circular_mean()]
        self.measurements = [np.NaN]
        self.grover_depths = [np.NaN]

    def update(
        self, measurement: int, grover_depth: int, show_update: bool = True
    ) -> None:
        """Update the posterior by one step.

        This currently assumes a VM prior, and approximates the
        posterior by a VM.

        Parameters
        ----------
        measurement : int, {0,1}
            Measurement value observed
        grover_depth : int
            Depth of the grover circuit used to obtain the measurement
        show_update : bool, optional
            Whether to display the update to the console, by default
            True
        """
        self.measurements.append(measurement)
        self.grover_depths.append(grover_depth)
        sign = (-1) ** measurement

        lambda_t = 4 * grover_depth + 2
        mu_t_prev, kappa_t_prev = self.estimated_params[self.t]

        bessel_vals = modified_bessel(
            [0, 1, lambda_t - 1, lambda_t, lambda_t + 1], float(kappa_t_prev)
        )
        denom = bessel_vals[0] + sign * bessel_vals[3] * np.cos(
            lambda_t * mu_t_prev
        )
        numer = np.exp(1j * mu_t_prev) * (
            bessel_vals[1]
            + sign
            * 0.5
            * (
                np.exp(1j * lambda_t * mu_t_prev) * bessel_vals[4]
                + np.exp(-1j * lambda_t * mu_t_prev) * bessel_vals[2]
            )
        )
        posterior_moment = numer / denom

        self.t += 1
        self.posterior_moments.append(posterior_moment)
        self.estimated_params.append(
            VonMises.generate_parameters_from_m1(posterior_moment)
        )

        if show_update:
            print(
                f"At t = {self.t}, the new estimated mean is "
                f"{self.estimated_params[-1][0]} and the new estimated "
                f"kappa is {self.estimated_params[-1][1]}"
            )

    def fixed_sequence_update(
        self, measurements: np.ndarray, schedule: BaseSamplingSchedule
    ):
        """Update the posterior distribution from a sampling schedule.

        Parameters
        ----------
        measurements : np.ndarray
            Measurements taken
        schedule : BaseSamplingSchedule
            Schedule the measurements are taken at
        """
        total_measurements = 0
        for i_grover_depth, i_n_shots in schedule.get_sampling_schedule():
            for j_shot in range(i_n_shots):
                self.update(measurements[i_n_shots + j_shot], i_grover_depth)
            total_measurements += i_n_shots

    def get_p_t(self, t: int) -> float:
        """Get the Bernoulli rv probability at time t.

        Parameters
        ----------
        t : int
            Time t to get the probability at

        Returns
        -------
        float
            Probability of seeing a 1 at time t
        """
        return 0.5 * (
            1
            - np.cos(
                (4 * self.grover_depths[t] + 2)
                * self.estimated_params[t - 1][0]
            )
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
            Locaiton parameter of prior von-Mises distribution

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

    def get_params(self, t: int = None) -> Tuple[float, float]:
        """Get the estimated parameters at time t.

        Parameters
        ----------
        t : int, optional
            Time to retrieve estimated parameters at, by default current
            time

        Returns
        -------
        Tuple[float,float]
            Pair of kappa, mu estimated at time t
        """
        if t:
            return self.estimated_params[t]
        else:
            return self.estimated_params[self.t]

    def expected_radius(self, grover_depth: int, t: int = None) -> float:
        """Calculate the expected radius of the next posterior.

        Parameters
        ----------
        grover_depth : int
            Grover depth to estimate radius for
        t : int, optional
            Time step to estimate the radius for, by default current
            time

        Returns
        -------
        float
            _description_
        """
        lamda = 4 * grover_depth + 2
        return self.expected_lambda_radius(lamda, t)

    def expected_lambda_radius(self, lamda: int, t: int = None) -> float:
        """Calculate the expected radius of the next step.

        See report for calculations.

        Parameters
        ----------
        lamda : int
            Defines the probability of observing a 1 as
            0.5(1-cos(lambda * theta))
        t : int, optional
            Time step to calculate the next radius for, by default use
            the current timestep

        Returns
        -------
        float
            Expected radius of the next step
        """
        mu, kappa = self.get_params(t)
        kappa = float(kappa)
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

    def eval_radii(self, max_depth: int) -> np.ndarray:
        """Evaluate the expected radius up to a maximum Grover depth.

        Parameters
        ----------
        max_depth : int
            Maximum depth to evaluate Grover depth to

        Returns
        -------
        np.ndarray
            Expected radius for values of lambda up to the given max
            depth
        """
        return np.ndarray(
            [self.expected_radius(i_d) for i_d in range(max_depth)]
        )
