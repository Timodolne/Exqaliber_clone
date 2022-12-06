"""Full process for amplitude estimation."""
from exqaliber.bayesian_updates.bayesian_model import BayesianModel
from exqaliber.bayesian_updates.distributions.normal import Normal
from exqaliber.bayesian_updates.distributions.von_mises import VonMises
from exqaliber.circuit_sampling.classical import (
    ClassicalAmplitudeEstimation,
)
from exqaliber.sampling_schedule.fixed_sampling_schedule.lis import (
    LinearIncrementalSequence,
)


def prior_von_mises(mu: float = 0.2, kappa: float = 0.6):
    """Generate a von Mises prior.

    Parameters
    ----------
    mu : float, optional
        Mean of the von Mises distribution, by default 0
    kappa : float, optional
        Concentration parameter, by default 0.6

    Returns
    -------
    VonMises :
        von Mises distribution
    """
    return VonMises(mu, kappa)


def prior_normal(mu: float, var: float):
    """Generate a normal prior.

    Parameters
    ----------
    mu : float
        Mean of the normal distribution
    var : float
        Varinace of the normal distribution
    """
    return Normal(mu, var)


if __name__ == "__main__":
    dist = prior_von_mises()
    schedule = LinearIncrementalSequence(10, 10)
    circuit_sampling = ClassicalAmplitudeEstimation(0.2)
    model = BayesianModel(dist)

    sample = circuit_sampling.sample_amplitude_estimation_predefined_schedule(
        schedule, True
    )
    model.fixed_sequence_update(sample, schedule)
