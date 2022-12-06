"""Implementation of a classical simulation of amplitude estimation."""
import numpy as np

from exqaliber.circuit_sampling.noise_model.base import (
    NOISE_MODEL,
    BaseNoiseModel,
)
from exqaliber.circuit_sampling.noise_model.depolarising import (
    Depolarising,
)
from exqaliber.circuit_sampling.noise_model.noiseless import Noiseless
from exqaliber.sampling_schedule.fixed_sampling_schedule.base import (
    BaseSamplingSchedule,
)


class ClassicalAmplitudeEstimation:
    r"""Classical simulation of amplitude estimation circuit.

    Attributes
    ----------
    theta : float
        theta such that A |0> = sin(theta)|psi_1>|1> +
        cos(theta)|psi_0>|0>
    """

    def __init__(
        self, theta: float, noise_model: BaseNoiseModel = Noiseless()
    ):
        r"""Initialise ClassicalAmplitudeEstimation.

        Parameters
        ----------
        theta : float
            theta such that A |0> = sin(theta)|psi_1>|1> +
            cos(theta)|psi_0>|0>
        noise_model : BaseNoiseModel, optional
            Noise model for the circuit. Default is Noiseless

        """
        self.theta = theta
        self.__noise_model = noise_model

    def sample(self, n_shots: int, alpha: float) -> np.ndarray:
        r"""Sample from amplitude / phase estimation type circuit.

        Parameters
        ----------
        n_shots : int
            Number of shots for the simulated circuit
        alpha : float
            Defines probability of observing 1 as
            (1 - cos(2*alpha * theta))/2 = sin^2(alpha * theta)

        Returns
        -------
        int : Number of 1's observed

        """
        return np.random.binomial(
            n_shots,
            0.5 * (np.ones(alpha.size) - np.cos(2 * alpha * self.theta)),
        )

    def sample_with_noise(self, n_shots: int, alpha: float):
        r"""Sample from amplitude estimation circuit with noise model.

        Parameters
        ----------
        n_shots : int
            Number of shots for the simulated circuit
        alpha: float
            Defines probability of noiseless circuit observing 1 as
            (1 - cos(2*alpha * theta))/2

        Returns
        -------
        int : Number of 1's observed

        Raises
        ------
        NotImplementedError :
            Noise model not implemented for this model
            Currently implemented: Noiseless, Depolarising

        """
        match self.__noise_model.get_type():
            case NOISE_MODEL.NOISELESS:
                sample = self.sample(n_shots, alpha)
            case NOISE_MODEL.DEPOLARISING:
                assert isinstance(self.__noise_model, Depolarising)

                n_decohered_shots = np.random.binomial(
                    n_shots, self.__noise_model.get_decoherence_probability()
                )
                n_coherent_shots = n_shots - n_decohered_shots

                sample = self.sample(
                    n_coherent_shots, alpha
                ) + np.random.binomial(n_decohered_shots, 0.5)
            case _:
                raise NotImplementedError(
                    f"{self.__noise_model.get_type()} Noise model not "
                    "currently implemented for classical sampling"
                )
        return sample

    def sample_fixed_schedule(
        self, schedule: np.ndarray, use_noise_model: bool = True
    ):
        r"""Sample for a fixed schedule."""
        pass

    def sample_amplitude_estimation_predefined_schedule(
        self, schedule: BaseSamplingSchedule, ordered: bool = False
    ):
        r"""Sample  amplitude estimation circuit with a given schedule.

        Parameters
        ----------
        schedule : BaseSamplingSchedule
            A schedule for pairs (m_i, n_i) of applications of
            controlled unitary and number of shots respectively.
        ordered: bool, optional
            Get the measurements as a sequence of 1's and 0's ordered by
            the given schedule, by default False

        Returns
        -------
        np.ndarray(shape=(schedule.length())) :
            Number of 1's for each noisy sample corresponding to the
            given schedule.

        """
        sample = self.sample_with_noise(
            schedule.get_n_shots_schedule(),
            2 * schedule.get_grover_depth_schedule()
            + np.ones(schedule.get_grover_depth_schedule().shape),
        )

        if ordered:
            idx = 0
            ordered_sample = np.zeros(schedule.get_n_shots_schedule().sum())

            for i_ones, i_n_shots in zip(
                sample, schedule.get_n_shots_schedule()
            ):
                i_sample = np.concatenate(
                    (np.ones(i_ones), np.zeros(i_n_shots - i_ones))
                )
                np.random.shuffle(i_sample)
                ordered_sample[idx : idx + i_n_shots] = i_sample
                idx += i_n_shots

            sample = ordered_sample.astype(int)

        return sample

    def set_noise_model(self, noise_model: BaseNoiseModel) -> None:
        r"""Set noise model.

        Parameters
        ----------
        noise_model: BaseNoiseModel
            Noise model to set

        """
        self.__noise_model = noise_model
