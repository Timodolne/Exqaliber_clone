"""Graph the behaviour of Exqaliber Amplitude Estimation."""
import pickle
import sys
from fractions import Fraction

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
from scipy.stats import norm
from tqdm import tqdm

from exqaliber.exqaliber_amplitude_estimation import (
    ExqaliberAmplitudeEstimation,
    ExqaliberAmplitudeEstimationResult,
)


def format_with_pi(
    x: float,
    tick_number: int = 0,
    max_denominator: int = 100,
    format_string: str = "",
):
    """Format a number as a fraction of pi when possible."""
    if x < 1e-8:
        return f"{x:{format_string}}"
    frac = Fraction(x / np.pi).limit_denominator(max_denominator)

    # check if the fraction matches
    if np.abs(x - frac * np.pi) < 1e-8:
        # whole fraction
        if frac.numerator == 1:
            if frac.denominator == 1:
                return r"$\pi$"
            return rf"$\pi/{frac.denominator}$"
        # multiple of pi
        elif frac.denominator == 1:
            return rf"${frac.numerator}\pi$"
        # combined fraction
        else:
            return rf"${frac.numerator}\pi/{frac.denominator}$"
    # not a fraction of pi
    else:
        return f"{x:{format_string}}"


def animate_exqaliber_amplitude_estimation(
    result: ExqaliberAmplitudeEstimationResult,
    experiment: dict,
    save: str = False,
    show: bool = True,
):
    """Animate the algorithm convergence over iterations."""
    distributions = result.distributions
    n_iter = len(result.powers)

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
        rf"Experiment: $\theta$: {format_with_pi(experiment['true_theta'])}, "
        rf"{mu_hat_str}: {format_with_pi(experiment['prior_mean'])}, "
        rf"{sigma_hat_str}: {format_with_pi(experiment['prior_std'])}. "
        rf"Method: {experiment['method']}"
    )
    fig.suptitle(title)

    plt.tight_layout()

    if save:
        anim.save(save, fps=10, extra_args=["-vcodec", "libx264"])

    if show:
        plt.show()


def convergence_plot(
    result: ExqaliberAmplitudeEstimationResult,
    experiment: dict,
    save: str = False,
    show: bool = True,
):
    """Plot the convergence of the algorithm."""
    distributions = result.distributions
    n_iter = len(result.powers)

    # create figure
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
        rf"Experiment: $\theta$: {format_with_pi(experiment['true_theta'])}, "
        rf"{mu_hat_str}: {format_with_pi(experiment['prior_mean'])}, "
        rf"{sigma_hat_str}: {format_with_pi(experiment['prior_std'])}. "
        rf"Method: {experiment['method']}"
    )
    fig.suptitle(title)

    plt.tight_layout()

    if save:
        plt.savefig(save)

    if show:
        plt.show()


def circular_histogram(
    results_multiple_thetas: list,
    theta_range: np.ndarray,
    experiment: dict,
    save: bool = False,
    show: bool = True,
):
    """Plot the circular histogram of nb of queries."""
    # get queries
    queries = np.array(
        [
            [res.num_oracle_queries for res in result]
            for result in results_multiple_thetas
        ]
    )
    mean_queries = queries.mean(axis=1)
    nb_reps = len(queries[0])

    # figure
    plt.figure(figsize=(10, 10), dpi=150)
    ax = plt.subplot(projection="polar")

    width = 2 * np.pi / resolution
    ax.bar(theta_range, mean_queries, width=width)

    # Plot title
    mu_hat_str = r"$\hat{\mu}$"
    sigma_hat_str = r"$\hat{\sigma}^2$"
    title = (
        f"Mean number (over {nb_reps} samples) of iterations"
        "before convergence.\n"
        rf"Experiment: {mu_hat_str}: "
        rf"{format_with_pi(experiment['prior_mean'])}, "
        rf"{sigma_hat_str}: {format_with_pi(experiment['prior_std'])}. "
        rf"Method: {experiment['method']}"
    )
    plt.title(title)

    plt.tight_layout()

    if save:
        plt.savefig(save)

    if show:
        plt.show()


def accuracy_plot_linear(
    results_multiple_thetas: list,
    theta_range: np.ndarray,
    experiment: dict,
    save: bool = False,
    show: bool = True,
):
    """Plot accuracy of estimation and theta."""
    estimations = np.array(
        [
            [res.estimation for res in result]
            for result in results_multiple_thetas
        ]
    )
    mean_estimations = estimations.mean(axis=1)
    nb_reps = len(estimations[0])

    # figure
    plt.figure(dpi=150)
    ax = plt.subplot()

    ax.plot(theta_range, mean_estimations)

    # X-axis
    ax.set_xlabel(r"$\theta$")
    ax.set_xlim(min(theta_range), max(theta_range))

    ax.xaxis.set_major_locator(plt.MultipleLocator(np.pi / 2))
    ax.xaxis.set_minor_locator(plt.MultipleLocator(np.pi / 4))
    ax.xaxis.set_major_formatter(plt.FuncFormatter(format_with_pi))

    # Y-axis
    ax.set_ylabel(r"Estimated $\mu$")
    ax.set_ylim(0, np.pi / 2)

    ax.yaxis.set_major_locator(plt.MultipleLocator(np.pi / 4))
    ax.yaxis.set_minor_locator(plt.MultipleLocator(np.pi / 8))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(format_with_pi))

    ax.grid(True)

    # Plot title
    mu_hat_str = r"$\hat{\mu}$"
    sigma_hat_str = r"$\hat{\sigma}^2$"
    title = (
        f"Mean estimation (over {nb_reps} samples) of "
        r"$\theta$ vs. actual $\theta$."
        f"\nExperiment: {mu_hat_str}: "
        rf"{format_with_pi(experiment['prior_mean'])}, "
        rf"{sigma_hat_str}: {format_with_pi(experiment['prior_std'])}. "
        rf"Method: {experiment['method']}"
    )
    plt.title(title)

    plt.tight_layout()

    if save:
        plt.savefig(save)

    if show:
        plt.show()


def run_experiment_one_theta(theta, experiment):
    """Run Exqaliber AE for one theta."""
    # set experiment
    experiment["true_theta"] = theta

    # do the experiment
    ae = ExqaliberAmplitudeEstimation(0.01, 0.01, **EXPERIMENT)
    result_one_theta = ae.estimate(None)

    print(f"Executed {len(result_one_theta.powers)} rounds")
    print(
        f"Finished with variance of {result_one_theta.variance:.6f} "
        f"and mean {result_one_theta.estimation:.6f}, "
        f"(true theta: {EXPERIMENT['true_theta']})."
    )

    return result_one_theta


def run_experiment_multiple_thetas(theta_range, experiment):
    """Create results for Exqaliber AE for multiple input thetas."""
    # recording
    results_multiple_thetas = []

    for theta in tqdm(theta_range, desc="theta", position=0, file=sys.stdout):
        results_theta = []
        experiment["true_theta"] = theta

        for i in tqdm(
            range(reps),
            desc=" repetitions",
            position=1,
            file=sys.stdout,
            leave=False,
        ):
            # tqdm.write(f'Starting repetition: {i: 2d}.\n', end='')
            ae = ExqaliberAmplitudeEstimation(0.01, 0.01, **EXPERIMENT)

            result = ae.estimate(None)

            results_theta.append(result)

        results_multiple_thetas.append(results_theta)

    return results_multiple_thetas


if __name__ == "__main__":

    format_with_pi(np.pi)

    # saving and running parameters
    run_or_load = "load"
    save_results = True
    show_results = True
    one_theta_experiment = False
    sweep_experiment = True

    # parameters all experiments
    prior_mean = np.pi / 4
    prior_std = 0.5
    method = "greedy"
    EXPERIMENT = {
        "prior_mean": prior_mean,
        "prior_std": prior_std,
        "method": method,
    }

    # parameters one run experiment
    true_theta = 1.0
    do_animation_plot = True
    do_convergence_plot = True

    # parameters theta sweep
    reps = 50
    resolution = 120
    theta_range = np.linspace(0, 2 * np.pi, resolution, endpoint=False)
    do_circular_histogram = True
    do_accuracy_plot_linear = True

    if one_theta_experiment:
        result_one_theta = run_experiment_one_theta(true_theta, EXPERIMENT)

        if do_animation_plot:
            filename = "results/animation.mp4" if save_results else False
            animate_exqaliber_amplitude_estimation(
                result_one_theta,
                experiment=EXPERIMENT,
                save=filename,
                show=show_results,
            )

        if do_convergence_plot:
            filename = "results/convergence.png" if save_results else False
            convergence_plot(
                result_one_theta,
                experiment=EXPERIMENT,
                save=filename,
                show=show_results,
            )

    if sweep_experiment:
        if run_or_load == "run":
            results_multiple_thetas = run_experiment_multiple_thetas(
                theta_range, experiment=EXPERIMENT
            )

            # save results
            with open("results/results_multiple_thetas.pkl", "wb") as f:
                pickle.dump(results_multiple_thetas, f)
            with open("results/theta_range.pkl", "wb") as f:
                pickle.dump(theta_range, f)

        elif run_or_load == "load":
            # load results
            with open("results/results_multiple_thetas.pkl", "rb") as f:
                results_multiple_thetas = pickle.load(f)
            with open("results/theta_range.pkl", "rb") as f:
                theta_range = pickle.load(f)

        if do_circular_histogram:
            filename = (
                "results/circular_histogram.png" if save_results else False
            )
            circular_histogram(
                results_multiple_thetas,
                theta_range,
                experiment=EXPERIMENT,
                save=filename,
                show=show_results,
            )

        if do_accuracy_plot_linear:
            filename = "results/accuracy_linear.png" if save_results else False
            accuracy_plot_linear(
                results_multiple_thetas,
                theta_range,
                experiment=EXPERIMENT,
                save=filename,
                show=show_results,
            )

    print("Done.")
