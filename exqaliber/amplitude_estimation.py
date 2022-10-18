"""Full process for amplitude estimation."""

from bayesian_updates.distributions.von_mises import VonMises
from circuit_sampling.classical import ClassicalAmplitudeEstimation
from sampling_schedule.fixed_sampling_schedule.lis import (
    LinearIncrementalSequence,
)


def prior_dist(mu: float = 0.2, kappa: float = 0.5):
    """Generate a prior distribution.

    Parameters
    ----------
    mu : float, optional
        Mean of the von Mises distribution, by default 0
    kappa : float, optional
        Concentration parameter, by default 0.5

    Returns
    -------
    VonMises :
        von Mises distribution
    """
    return VonMises(mu, kappa)


if __name__ == "__main__":
    dist = prior_dist()
    schedule = LinearIncrementalSequence(10, 10)
    circuit_sampling = ClassicalAmplitudeEstimation(dist.get_circular_mean())
    print(dist.sample())
    print(schedule.get_sampling_schedule())
    print(
        circuit_sampling.sample_amplitude_estimation_predefined_schedule(
            schedule
        )
    )
