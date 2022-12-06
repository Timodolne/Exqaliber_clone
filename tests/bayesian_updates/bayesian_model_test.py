import pytest

from exqaliber.bayesian_updates.bayesian_model import BayesianModel
from exqaliber.bayesian_updates.distributions.von_mises import VonMises


@pytest.mark.parametrize(
    "lamda, kappa, mu, prob",
    [
        (0, 2, 1, 0),
        (0, 5, 1, 0),
        (0, 2, 4, 0),
        (0, 5, 4, 0),
        (1, 0.5, 1, 0.43448845007523108252),
        (20, 50, -1, 0.4962135630356859244),
    ],
)
def test_ber_prob(lamda, kappa, mu, prob):
    assert BayesianModel.get_ber_prob(lamda, kappa, mu) == pytest.approx(
        prob, rel=1e-9
    )


def gen_bayesian_model(mu, kappa):
    return BayesianModel(VonMises(mu, kappa))


@pytest.mark.parametrize(
    "mu, kappa, measurement, grover_depth, loc, scale",
    [
        (0, 0.5, 0, 0, 0, 0.3543616284547993661),
        (0, 0.5, 1, 0, 0, 0.1237179282783207305),
        (0, 0.5, 0, 1, 0, 0.2425034064549478046),
        (0, 0.5, 1, 1, 0, 0.2424958187042150995),
        (0, 2, 0, 1, 0, 0.699488064810048570985),
        (0, 2, 1, 1, 0, 0.6960588439486131781),
        (0, 10, 0, 1, 0, 0.9735304316107326173),
        (0, 10, 1, 1, 0, 0.9142022585045977153),
        (0, 10, 0, 10, 0, 0.9485998259548459589),
        (0, 10, 1, 10, 0, 0.9485998259548459589),
    ],
)
class TestUpdate:
    @staticmethod
    def test_update_loc(mu, kappa, measurement, grover_depth, loc, scale):
        model = gen_bayesian_model(mu, kappa)
        model.update(measurement, grover_depth, False)
        assert model.get_params()[0] == pytest.approx(loc, rel=1e-9)

    @staticmethod
    def test_update_scale(mu, kappa, measurement, grover_depth, loc, scale):
        model = gen_bayesian_model(mu, kappa)
        model.update(measurement, grover_depth, False)
        assert abs(model.posterior_moments[-1]) == pytest.approx(
            scale, rel=1e-9
        )
