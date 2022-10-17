"""Full process for amplitude estimation."""

from bayesian_updates.approximate_posterior.circular_distributions.von_mises import VonMises
from sampling_schedule.fixed_sampling_schedule.lis import LinearIncrementalSequence
from circuit_sampling.classical import ClassicalAmplitudeEstimation

def prior_dist(mu: float = 0, kappa: float = 0.5):
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
    schedule = LinearIncrementalSequence(10,10)
    circuit_sampling = ClassicalAmplitudeEstimation(0)
    print(dist.sample())
    print(schedule.get_sampling_schedule())
    print(circuit_sampling.sample_amplitude_estimation_predefined_schedule(schedule))
