import numpy as np
import pytest
from scipy import stats

from exqaliber.bayesian_updates.distributions.circular_distribution_base import (
    CIRCULAR_DISTRIBUTION,
)
from exqaliber.bayesian_updates.distributions.von_mises import VonMises


def test_distribution_type():
    assert VonMises(0.2, 0.6).get_type() == CIRCULAR_DISTRIBUTION.VON_MISES


@pytest.mark.parametrize(
    "valid_location, non_positive_kappa", [(0.2, -0.4), (0.6, 0)]
)
def test_non_positive_kappa(valid_location, non_positive_kappa):
    with pytest.raises(ValueError):
        VonMises(valid_location, non_positive_kappa)


@pytest.mark.parametrize(
    "invalid_location, valid_scale", [(-3.5, 0.1), (np.pi, 5), (5, 2)]
)
def test_out_of_bounds_location(invalid_location, valid_scale):
    with pytest.warns(UserWarning):
        VonMises(invalid_location, valid_scale)


@pytest.mark.parametrize(
    "value, order, ratio, n_steps",
    [
        (0.5, 0, 0.24249961258080, 10),
        (10, 0, 0.948599825954845, 20),
        (0.001, 0, 0.00049999993750, 10),
        (1, 4, 0.0991783823997125, 10),
    ],
)
def test_bessel_ratio(value, order, ratio, n_steps):
    assert VonMises.get_bessel_ratio(value, order, n_steps) == pytest.approx(
        ratio, rel=1e-9
    )


@pytest.mark.parametrize(
    "valid_location, valid_scale",
    [(0, 0.4), (-np.pi / 2, 2), (-0.2, 10), (1.4, 0.006)],
)
class TestValidDistribution:
    def test_get_loc(self, valid_location, valid_scale):
        dist = VonMises(valid_location, valid_scale)
        assert dist.generate_parameters_from_m1(dist.get_circular_mean())[
            0
        ] == pytest.approx(valid_location, rel=1e-9)

    def test_invert_bessel_ratio(self, valid_location, valid_scale):
        dist = VonMises(valid_location, valid_scale)
        assert dist.get_vm_concentration_param(
            dist.get_circular_mean()
        ) == pytest.approx(valid_scale, rel=1e-9)

    def test_get_mu(self, valid_location, valid_scale):
        dist = VonMises(valid_location, valid_scale)
        assert dist.get_parameters()["mu"] == pytest.approx(
            valid_location, rel=1e-9
        )

    def test_get_kappa(self, valid_location, valid_scale):
        dist = VonMises(valid_location, valid_scale)
        assert dist.get_parameters()["kappa"] == pytest.approx(
            valid_scale, rel=1e-9
        )

    def test_sample(self, valid_location, valid_scale):
        sample = VonMises(valid_location, valid_scale).sample(100)
        assert (
            stats.kstest(
                sample, stats.vonmises.cdf, (valid_scale, valid_location)
            )[1]
            < 0.95
        )
