"""Bayesian update model."""
from typing import Tuple, Union

import numpy as np

from exqaliber.sampling_schedule.fixed_sampling_schedule.base import (
    BaseSamplingSchedule,
)

from .distributions.normal import Normal
from .distributions.von_mises import VonMises


class BayesianModel:
    """Bayesian model for sequential estimation.

    Attributes
    ----------
    t : int
        Number of measurements taken so far
    prior: Union[VonMises, Normal]
        Prior distribution for theta
    estimated_params: List[Tuple[float,float]]
        Estimated parameters for (mu, kappa) at each time step
    measurements: List[int]
        Measurement at time t
    grover_depths: List[int]
        Grover depth at time t
    """

    def __init__(self, prior: Union[VonMises, Normal]) -> None:
        """Initialise BayesianModel.

        Parameters
        ----------
        prior : VonMises, Normal
            Prior distribution for the Bayesian updates
        """
        self.t = 0
        self.prior = prior

        self.estimated_params = [tuple(prior.get_parameters().values())]

        if isinstance(prior, VonMises):
            self.posterior_moments = [prior.get_circular_mean()]
        elif isinstance(prior, Normal):
            self.posterior_moments = [tuple(prior.get_parameters().values())]
        else:
            raise NotImplementedError(
                "Bayesian model not implemented for given prior"
            )

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

        current_lambda = 4 * grover_depth + 2
        prev_loc, prev_scale = self.estimated_params[self.t]

        posterior_moment = self.prior.update(
            measurement, current_lambda, prev_loc, prev_scale
        )

        self.t += 1
        self.posterior_moments.append(posterior_moment)

        if isinstance(self.prior, VonMises):
            self.estimated_params.append(
                VonMises.generate_parameters_from_m1(posterior_moment)
            )
        else:
            self.estimated_params.append(posterior_moment)

        if show_update:
            print(
                f"At t = {self.t}, the new estimated mean is "
                f"{self.estimated_params[-1][0]} and the new estimated "
                f"scale is {self.estimated_params[-1][1]}"
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
