"""Graph the behaviour of Exqaliber Amplitude Estimation."""
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
from scipy.stats import norm

from exqaliber.exqaliber_amplitude_estimation import (
    ExqaliberAmplitudeEstimation,
    ExqaliberAmplitudeEstimationResult,
)


def animate_exqaliber_amplitude_estimation(
    result: ExqaliberAmplitudeEstimationResult,
    experiment: dict,
    save: str = False,
):
    """Animate the algorithm convergence over iterations."""
    distributions = result.distributions

    # First set up figure, axis, and plot element we want to animate
    fig, axs = plt.subplots(2, 1, figsize=(5, 10))
    (curve,) = axs[1].plot([], [], lw=2)
    (mean,) = axs[0].plot([], [], lw=2)
    lines = [curve, mean]

    axs[0].set_xlabel(r"Estimation for $\theta$")
    axs[0].set_ylabel(r"Iteration")

    axs[1].set_xlabel(r"$\theta$")
    axs[1].set_ylabel("Density")

    # Plot true and prior theta
    axs[0].vlines(
        x=EXPERIMENT["true_theta"],
        ymin=-n_iter,
        ymax=n_iter,
        label=r"True $\theta$",
    )
    axs[1].vlines(
        x=EXPERIMENT["true_theta"], ymin=0, ymax=15, label=r"True $\theta$"
    )

    axs[0].vlines(
        x=EXPERIMENT["prior_mean"],
        ymin=-n_iter,
        ymax=n_iter,
        label=r"Prior $\theta$",
        linestyles="--",
    )
    axs[1].vlines(
        x=EXPERIMENT["prior_mean"],
        ymin=0,
        ymax=15,
        label=r"Prior $\theta$",
        linestyles="--",
    )

    xmin = -np.pi
    xmax = np.pi
    axs[0].set_xlim(xmin, xmax)
    axs[1].set_xlim(xmin, xmax)
    axs[1].set_ylim(0, 15)

    axs[0].legend(loc="upper right")
    axs[1].legend(loc="upper right")

    # initialization function: plot the background of each frame
    def init():
        """Initialise the plots."""
        lines[0].set_data([], [])
        lines[1].set_data([1], [1])
        return lines

    x = np.linspace(-np.pi, np.pi, 1000)

    # animation function.  This is called sequentially
    def animate(frame, *fargs):
        """Animate the frame of the plots."""
        dist = distributions[frame]
        rv = norm(dist.mean, dist.standard_deviation)
        y = rv.pdf(x)

        means = [dist.mean for dist in distributions[:frame]]
        ymin = frame
        ymax = frame - n_iter
        axs[0].set_ylim(ymin, ymax)

        # Hide negative ticks
        yticks = axs[0].get_yticks()
        for i, tick in enumerate(yticks):
            if tick < 0:
                axs[0].yaxis.get_major_ticks()[i].label1.set_visible(False)
                axs[0].yaxis.get_major_ticks()[i].tick1line.set_visible(False)
                axs[0].yaxis.get_major_ticks()[i].tick2line.set_visible(False)
            else:
                axs[0].yaxis.get_major_ticks()[i].label1.set_visible(True)
                axs[0].yaxis.get_major_ticks()[i].tick1line.set_visible(True)
                axs[0].yaxis.get_major_ticks()[i].tick2line.set_visible(True)

        lines[0].set_data(x, y)
        lines[1].set_data(means, range(frame))

        return lines

    # call the animator.
    anim = animation.FuncAnimation(
        fig,
        animate,
        init_func=init,
        frames=n_iter,
        interval=40,
        blit=False,
    )

    # Plot title
    mu_hat_str = r"$\hat{\mu}$"
    sigma_hat_str = r"$\hat{\sigma}^2$"
    title = (
        rf"Experiment: $\theta$: {experiment['true_theta']}, "
        rf"{mu_hat_str}: {experiment['prior_mean']}, "
        rf"{sigma_hat_str}: {experiment['prior_std']}. "
        rf"Method: {experiment['method']}"
    )
    fig.suptitle(title)

    plt.tight_layout()

    if save:
        anim.save(save, fps=10, extra_args=["-vcodec", "libx264"])

    plt.show()


def convergence_plot(
    result: ExqaliberAmplitudeEstimationResult,
    experiment: dict,
    save: str = False,
):
    """Plot the convergence of the algorithm."""
    distributions = result.distributions
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))

    # Variance plot
    x = range(n_iter)
    y = [np.log(dist.standard_deviation) for dist in distributions]
    axs[0].plot(x, y)

    axs[0].set_xlim(0, n_iter)

    axs[0].set_xlabel("Iteration")
    axs[0].set_ylabel(r"$\log\sigma^2$")

    axs[0].set_title("Log variance")

    # Mean plot
    y = [dist.mean for dist in distributions]
    axs[1].plot(x, y, label=r"$\mu$")
    axs[1].hlines(
        y=experiment["prior_mean"],
        xmin=0,
        xmax=n_iter,
        label=r"Prior $\theta$",
        linestyles="dotted",
    )
    axs[1].hlines(
        y=experiment["true_theta"],
        xmin=0,
        xmax=n_iter,
        label=r"True $\theta$",
        linestyles="--",
    )

    axs[1].set_xlim(0, n_iter)
    axs[1].set_ylim(-np.pi, np.pi)

    axs[1].set_xlabel("Iteration")
    axs[1].set_ylabel(r"$\mu$")

    axs[1].legend()

    axs[1].set_title(r"Estimation of $\theta$")

    # Powers plot
    y = result.powers
    axs[2].plot(x, y)

    text_x = (1 / 10) * max(x)
    text_y = (5 / 6) * max(y)
    text = f"Total {sum(y)} oracle calls"
    axs[2].text(text_x, text_y, text)

    axs[2].set_xlim(0, n_iter)

    axs[2].set_xlabel("Iteration")
    axs[2].set_ylabel(r"$k$")

    axs[2].set_title(r"Oracle calls $k$")

    # Plot title
    mu_hat_str = r"$\hat{\mu}$"
    sigma_hat_str = r"$\hat{\sigma}^2$"
    title = (
        rf"Experiment: $\theta$: {experiment['true_theta']}, "
        rf"{mu_hat_str}: {experiment['prior_mean']}, "
        rf"{sigma_hat_str}: {experiment['prior_std']}. "
        rf"Method: {experiment['method']}"
    )
    fig.suptitle(title)

    plt.tight_layout()

    if save:
        plt.savefig(save)

    plt.show()


if __name__ == "__main__":

    EXPERIMENT = {
        "true_theta": 1.6,
        "prior_mean": 0.5,
        "prior_std": 0.5,
        "method": "greedy",
    }
    save = False

    ae = ExqaliberAmplitudeEstimation(0.01, 0.01, **EXPERIMENT)
    estimation_problem = None

    result = ae.estimate(estimation_problem)
    n_iter = len(result.powers)

    print(f"Executed {len(result.powers)} rounds")
    print(
        f"Finished with variance of {result.variance:.6f} "
        f"and mean {result.estimation:.6f}, "
        f"(true theta: {EXPERIMENT['true_theta']})."
    )

    filename = "animation.mp4" if save else False
    animate_exqaliber_amplitude_estimation(
        result, experiment=EXPERIMENT, save=filename
    )

    filename = "convergence.png" if save else False
    convergence_plot(result, experiment=EXPERIMENT, save=filename)

    print("Done.")
