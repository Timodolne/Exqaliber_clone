"""Graph the expected radius of the von Mises distribution."""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from exqaliber.bayesian_updates.bayesian_model import BayesianModel
from exqaliber.bayesian_updates.distributions.von_mises import VonMises


def graph_radius(
    max_lambda: int, mu: float = 0.2, kappa: float = 1000
) -> None:
    """Graph the expected radius of a von Mises distribution.

    Parameters
    ----------
    max_lambda : int
        Largest value of lambda to calculate the expected radius for.
        We take p(1) = 0.5(1 - cos(lambda * theta))
    mu : float, optional
        Location parameter of the von Mises distribution, by default 0.2
    kappa : float
        Scale parameter of the von Mises distribution, by default 1000
    """
    model = BayesianModel(VonMises(mu, kappa))

    lambdas = np.array([i + 1 for i in range(max_lambda)], dtype=int)
    radii = model.expected_lambda_radius(lambdas)

    df = pd.DataFrame(
        np.array((lambdas, radii)).T,
        columns=["Lambda", "Expected Radius"],
    )
    df["Evaluation"] = np.array(["Exact"] * max_lambda)
    sns.lineplot(df, x="Lambda", y="Expected Radius")
    plt.show()


if __name__ == "__main__":
    max_depth = 20
    max_lambda = 4 * max_depth + 2
    kappa = 1000
    mu = 0.2

    graph_radius(max_lambda, mu, kappa)
