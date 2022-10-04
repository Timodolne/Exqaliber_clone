import numpy as np

from exqaliber.sampling_schedule.fixed_sampling_schedule.base import \
    BaseSamplingSchedule
from exqaliber.circuit_sampling.noise_model.base import NOISE_MODEL, BaseNoiseModel
from exqaliber.circuit_sampling.noise_model.noiseless import Noiseless


class ClassicalAmplitudeEstimation():
    r'''Classical analytic simulation of amplitude (or phase) estimation circuit.

    Attributes
    ----------
    theta : float
        theta such that A |0> = sin(theta)|psi_1>|1> +
        cos(theta)|psi_0>|0>
    __noise_model : BaseNoiseModel
        Noise model for the circuit

    '''

    def __init__(self, theta: float, noise_model: BaseNoiseModel = Noiseless()):
        r"""
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
        r'''Sample from amplitude / phase estimation type circuit

        Parameters
        ----------
        n_shots : int
            Number of shots for the simulated circuit
        alpha : float
            Defines probability of observing 1 as (1 - cos(2*alpha * theta))/2 = sin^2(alpha * theta)

        Returns
        -------
        int : Number of 1's observed

        '''

        return np.random.binomial(n_shots, 0.5*(np.ones(alpha.size)- np.cos(2*alpha * self.theta)))

    def sample_with_noise(self, n_shots: int, alpha: float):
        r'''Sample from amplitude / phase estimation type circuit with noise model

        Parameters
        ----------
        n_shots : int
            Number of shots for the simulated circuit
        alpha: float
            Defines probability of noiseless circuit observing 1 as (1 - cos(alpha * theta))/2

        Returns
        -------
        int : Number of 1's observed

        Raises
        ------
        NotImplementedError :
            Noise model not implemented for this model
            Currently implemented: Noiseless, Depolarising

        '''

        match self.noise_model.get_type():
            case NOISE_MODEL.NOISELESS:
                return self.sample(n_shots, alpha)
            case NOISE_MODEL.DEPOLARISING:
                n_depolarised = np.random.binomial(n_shots, self.noise_model.get_p())
                return self.sample(n_shots - n_depolarised, alpha) + np.random.binomial(n_depolarised,0.5)
            case _:
                raise NotImplementedError("Noise model not currently implemented for classical sampling")

    def sample_fixed_schedule(self, schedule: np.ndarray, use_noise_model: bool = True):
        r'''Sample for a fixed schedule'''
        pass

    def sample_amplitude_estimation_predefined_schedule(self, schedule: BaseSamplingSchedule):
        r'''Sample from an amplitude estimation circuit with a given schedule

        Parameters
        ----------
        schedule : BaseSamplingSchedule
            A schedule for pairs (m_i, n_i) of applications of controlled unitary and number of shots
            respectively

        Returns
        -------
        np.ndarray(shape=(schedule.length())) :
            Number of 1's for each noisy sample corresponding to the given schedule

        '''

        return self.sample_with_noise(schedule.get_sampling_schedule()[:,1], 2*schedule.get_sampling_schedule()[:,0] +  np.ones(schedule.get_sampling_schedule()[:,0].shape))

    def set_noise_model(self, noise_model: BaseNoiseModel) -> None:
        r'''Set noise model

        Parameters
        ----------
        noise_model: BaseNoiseModel
            Noise model to set

        '''

        self.__noise_model = noise_model
