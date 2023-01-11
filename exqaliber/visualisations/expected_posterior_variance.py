"""Graph the expected posterior with Gaussian prior and posterior."""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from exqaliber.bayesian_updates.distributions.normal import Normal


def graph_expected_posterior_variance(
    max_lambda: int, mu: float = 0.2, sigma: float = 1
) -> None:
    """Graph the expected posterior variance with Gaussian prior.

    Parameters
    ----------
    max_lambda : int
        Largest value of lambda to calculate the expected radius for.
        We take p(1) = 0.5(1 - cos(lambda * mu))
    mu : float
        Location parameter of the current normal distribution
    sigma : float
        Scale parameter of the current normal distribution
    """
    lambdas = np.array([i + 1 for i in range(max_lambda)], dtype=int)
    variance_reduction_factors = Normal.eval_lambdas(lambdas, mu, sigma)
    variances = [
        sigma**2 * (1 - sigma**2 * V) for V in variance_reduction_factors
    ]

    df = pd.DataFrame(
        np.array((lambdas, variances)).T,
        columns=["Lambda", "Expected Posterior Variance"],
    )
    df["Evaluation"] = np.array(["Exact"] * max_lambda)
    sns.lineplot(df, x="Lambda", y="Expected Posterior Variance")
    plt.show()


if __name__ == "__main__":
    max_depth = 50
    max_lambda = 4 * max_depth + 2
    sigma = 0.01
    mu = 0.2

    graph_expected_posterior_variance(max_lambda, mu, sigma)
