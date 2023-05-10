"""Graph the behaviour of Exqaliber Amplitude Estimation."""
import hashlib
import os.path
import pickle
import queue
from fractions import Fraction
from functools import partial
from itertools import product
from multiprocessing import Pool, Queue

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
from matplotlib.widgets import Slider
from scipy.stats import norm

try:
    __IPYTHON__
    from tqdm.notebook import tqdm
except NameError:
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


def experiment_string(experiment, sweep=False):
    """Create a string for plot titles."""
    mu_hat_str = r"$\hat{\mu}$"
    sigma_hat_str = r"$\hat{\sigma}^2$"
    if experiment["prior_mean"] == "true_theta":
        prior_mean_str = r"$\theta$"
    elif experiment["prior_mean"] == "uniform":
        prior_mean_str = r"$U(0,\pi)$"
    elif experiment["prior_mean"] == "gaussian":
        prior_mean_str = r"$N(0,\pi)$"
    else:
        prior_mean_str = format_with_pi(experiment["prior_mean"])

    if not sweep:
        experiment = (
            rf"$\theta= ${format_with_pi(experiment['true_theta'])}, "
            rf"{mu_hat_str}$= ${prior_mean_str}, "
            rf"{sigma_hat_str}$= ${format_with_pi(experiment['prior_std'])}. "
            rf"$\epsilon = ${experiment['epsilon_target']}. "
            rf"Method: {experiment['method']}"
        )
    else:
        experiment = (
            rf"{mu_hat_str}$= ${prior_mean_str}, "
            rf"{sigma_hat_str}$= ${format_with_pi(experiment['prior_std'])}. "
            rf"$\epsilon = ${experiment['epsilon_target']}. "
            rf"Method: {experiment['method']}"
        )

    return experiment


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
        x=experiment["true_theta"],
        ymin=-n_iter,
        ymax=n_iter,
        label=r"True $\theta$",
        linestyles="--",
    )
    axs[1].vlines(
        x=experiment["true_theta"],
        ymin=0,
        ymax=15,
        label=r"True $\theta$",
        linestyles="--",
    )

    if experiment["prior_mean"] != "true_theta":
        axs[0].vlines(
            x=experiment["prior_mean"],
            ymin=-n_iter,
            ymax=n_iter,
            label=r"Prior $\theta$",
            linestyles="--",
        )
        axs[1].vlines(
            x=experiment["prior_mean"],
            ymin=0,
            ymax=15,
            label=r"Prior $\theta$",
            linestyles="--",
        )

    xmin = 0
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
        ymin = frame - 1
        ymax = frame - n_iter - 1
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
    title = f"Convergence animation.\n{experiment_string(experiment)}"
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
    if experiment["prior_mean"] != "true_theta":
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
    axs[1].set_ylim(0, np.pi)

    axs[1].set_xlabel("Iteration")
    axs[1].set_ylabel(r"$\mu$")

    axs[1].legend()

    axs[1].set_title(r"Estimation of $\theta$")

    # Powers plot
    y = 2 * np.array(result.powers) + 1
    axs[2].plot(x, y)

    text_x = (1 / 10) * max(x)
    text_y = (5 / 6) * max(y)
    text = f"Total {result.num_oracle_queries} oracle calls"
    axs[2].text(text_x, text_y, text)

    axs[2].set_xlim(0, n_iter)
    axs[2].set_yscale("log")

    axs[2].set_xlabel("Iteration")
    axs[2].set_ylabel(r"$\lambda$")

    axs[2].set_title(r"Oracle calls $k$")

    # Plot title
    title = f"Convergence.\n{experiment_string(experiment)}"
    fig.suptitle(title)

    plt.tight_layout()

    if save:
        plt.savefig(save)

    if show:
        plt.show()


def circular_histogram(
    results: dict,
    save: bool = False,
    show: bool = True,
    rules: dict = None,
    experiment: dict = None,
):
    """Plot the circular histogram of nb of queries."""
    if (
        isinstance(
            results["parameters"]["epsilon_target"],
            (list, tuple, set, np.ndarray),
        )
        and "epsilon_target" not in rules.keys()
    ):
        raise "Choose one epsilon target for circular histogram."

    if rules is None:
        rules = {"zeta": 0}
    # get queries
    results_sliced = get_results_slice(results, rules=rules)

    thetas = np.array([theta for (theta, i) in results_sliced.keys()])
    queries = np.array(
        [res.num_oracle_queries for res in results_sliced.values()]
    )

    # parameters
    nb_reps = results["parameters"]["reps"]
    theta_range = results["parameters"]["true_theta"]

    # figure
    fig = plt.figure(figsize=(7, 7), dpi=100)
    ax = plt.subplot(projection="polar")

    # data
    width = np.pi / (len(theta_range) - 1)
    max_magnitude_y = int(np.ceil(np.log10(max(queries))))
    max_y = 10 ** (max_magnitude_y)
    min_magnitude_y = int(np.floor(np.log10(min(queries))))

    x_bins = np.arange(-width / 2, np.pi + 3 * width / 2, width)
    y_bins = np.logspace(
        min_magnitude_y, max_magnitude_y, 2 * max_magnitude_y + 1
    )
    bins = [x_bins, y_bins]

    h = ax.hist2d(
        thetas.flatten(),
        queries.flatten(),
        bins=bins,
        cmin=1,
        cmax=nb_reps,
        norm=mpl.colors.LogNorm(),
    )

    # axis
    ax.set_xlim(0, np.pi)
    ax.set_rlim(1, 3 * max_y)
    ax.set_rscale("symlog")
    ax.grid(True)

    # Plot title
    title = "Number of iterations before convergence.\n"
    if experiment is not None:
        title += f"{experiment_string(experiment, True)}"
    plt.title(title)

    fig.colorbar(h[3], ax=ax, location="bottom", label="oracle queries")

    plt.tight_layout(pad=1.0)

    if save:
        plt.savefig(save, dpi=300)

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
    mean_estimations = estimations.mean(axis=1) % (2 * np.pi)
    nb_reps = len(estimations[0])

    # figure
    plt.figure(dpi=150)
    ax = plt.subplot()

    ax.plot((theta_range % (2 * np.pi)), mean_estimations)

    # X-axis
    ax.set_xlabel(r"$\theta$")
    ax.set_xlim(0, np.pi)

    ax.xaxis.set_major_locator(plt.MultipleLocator(np.pi / 4))
    ax.xaxis.set_minor_locator(plt.MultipleLocator(np.pi / 8))
    ax.xaxis.set_major_formatter(plt.FuncFormatter(format_with_pi))

    # Y-axis
    ax.set_ylabel(r"Estimated $\mu$")
    ax.set_ylim(0, np.pi)

    ax.yaxis.set_major_locator(plt.MultipleLocator(np.pi / 4))
    ax.yaxis.set_minor_locator(plt.MultipleLocator(np.pi / 8))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(format_with_pi))

    ax.grid(True)

    # Plot title
    title = (
        f"Mean estimation (over {nb_reps} samples) of "
        r"$\theta$ vs. actual $\theta$."
        f"\n{experiment_string(experiment, True)}"
    )
    plt.title(title)

    plt.tight_layout()

    if save:
        plt.savefig(save)

    if show:
        plt.show()


def error_in_estimate_2d_hist(
    results_multiple_thetas: list,
    theta_range: np.ndarray,
    experiment: dict,
    save: bool = False,
    show: bool = True,
):
    """Plot the error in the estimate."""
    estimations = np.array(
        [
            [res.estimation for res in result]
            for result in results_multiple_thetas
        ]
    )
    estimations = estimations % (2 * np.pi)
    thetas = theta_range % (2 * np.pi)

    nb_reps = len(estimations[0])

    thetas_repeated = np.vstack([thetas] * nb_reps).T
    errors = estimations - thetas_repeated

    # data
    width = np.pi / (len(theta_range) + 1)
    min_y = errors.min()
    max_y = errors.max()

    # build figure
    fig, axs = plt.subplots(2, 2, figsize=(12, 9))

    # Left figures
    val = thetas[2]

    epsilon_target = experiment.get("epsilon_target")

    # top figure
    bins_top = np.linspace(-3 * epsilon_target, 3 * epsilon_target, 100)
    i = np.argwhere(thetas == val)

    axs[0, 0].hist(errors[i].flatten(), bins=bins_top)

    # bottom figure
    bins_bottom = np.linspace(min_y, max_y, 100)
    axs[1, 0].hist(errors[i].flatten(), bins=bins_bottom)

    # Right figures
    # top figure
    x_bins = np.arange(-width / 2, np.pi + width / 2, width)
    y_bins = bins_top
    bins = [x_bins, y_bins]

    axs[0, 1].hist2d(
        thetas_repeated.flatten(),
        errors.flatten(),
        bins=bins,
        cmin=1,
        cmax=nb_reps,
        norm=mpl.colors.LogNorm(),
    )
    vline_top = axs[0, 1].axvline(
        val, ymin=-1, ymax=1, linestyle="--", color="red"
    )

    # bottom figure
    x_bins = np.arange(-width / 2, np.pi + width / 2, width)
    y_bins = bins_bottom
    bins = [x_bins, y_bins]

    h2 = axs[1, 1].hist2d(
        thetas_repeated.flatten(),
        errors.flatten(),
        bins=bins,
        cmin=1,
        cmax=nb_reps,
        norm=mpl.colors.LogNorm(),
    )
    vline_bottom = axs[1, 1].axvline(
        val, ymin=y_bins.min(), ymax=y_bins.max(), linestyle="--", color="red"
    )

    # X-axis right
    for ax in axs[:, 1]:
        ax.set_xlabel(r"$\theta$")
        ax.set_xlim(0, np.pi)

        ax.xaxis.set_major_locator(plt.MultipleLocator(np.pi / 4))
        ax.xaxis.set_minor_locator(plt.MultipleLocator(np.pi / 8))
        ax.xaxis.set_major_formatter(plt.FuncFormatter(format_with_pi))

    # Y-axes right
    axs[0, 1].set_ylim(-3 * epsilon_target, 3 * epsilon_target)
    axs[0, 1].set_ylabel("Error in estimate")

    axs[1, 1].set_ylim(min_y, max_y)
    axs[1, 1].set_ylabel("Error in estimate")

    # Axes left
    for ax in axs[:, 0]:
        ax.set_xlabel("Error in estimate")

        ax.set_ylim(1, None)
        ax.set_yscale("log")
        ax.set_ylabel("Frequency")

    axs[0, 0].set_title(rf"Errors in estimates for $\theta=${val:.3f}")

    # Finishing figure
    title = (
        f"Error in estimate of theta "
        f"\n{experiment_string(experiment, True)}"
    )
    plt.suptitle(title)

    # Colorbar
    # make room for colorbar and slider
    fig.subplots_adjust(bottom=0.1, right=0.95)
    fig.colorbar(h2[3], ax=axs[:, 1], label="Runs")

    # Slider

    valmin = thetas[0]
    valmax = thetas[-1]
    valstep = thetas[1] - thetas[0]

    x0 = axs[1, 1].get_position().x0
    x1 = axs[1, 1].get_position().x1 - x0

    slider_ax = fig.add_axes([x0, 0.03, x1, 0.03])
    slider = Slider(
        slider_ax,
        label=r"$\theta$",
        valmin=valmin,
        valmax=valmax,
        valstep=valstep,
        valinit=val,
    )

    # update function
    def update(val):
        # Left figures

        # top figure
        i = np.argwhere(thetas == val)
        axs[0, 0].cla()
        axs[0, 0].hist(errors[i].flatten(), bins=bins_top)

        axs[1, 0].cla()
        axs[1, 0].hist(errors[i].flatten(), bins=bins_bottom)

        # Axes left
        for ax in axs[:, 0]:
            ax.set_xlabel("Error in estimate")

            ax.set_ylim(1, None)
            ax.set_yscale("log")
            ax.set_ylabel("Frequency")

        axs[0, 0].set_title(rf"Errors in estimates for $\theta=${val:.3f}")

        # Right figures
        vline_bottom.set_xdata(val)
        vline_top.set_xdata(val)

    slider.on_changed(update)

    if save:
        plt.savefig(save, dpi=300)

    if show:
        plt.show()


def run_experiment_one_theta(theta, experiment):
    """Run Exqaliber AE for one theta."""
    # do the experiment
    ae = ExqaliberAmplitudeEstimation(**experiment)
    result_one_theta = ae.estimate(theta)

    print(f"Executed {len(result_one_theta.powers)} rounds")
    print(
        f"Finished with variance of {result_one_theta.variance:.6f} "
        f"and mean {result_one_theta.estimation:.6f}, "
        f"(true theta: {experiment['true_theta']})."
    )

    return result_one_theta


def run_single_experiment(experiment, output="sparse"):
    """Run one experiment (wrapper for multiprocessing)."""
    ae = ExqaliberAmplitudeEstimation(**experiment)
    max_iter = experiment["max_iter"]
    result = ae.estimate(
        experiment["true_theta"], output=output, max_iter=max_iter
    )

    return result


def run_experiment_single_rep(args, experiment_f=run_single_experiment):
    """Run a single repetition of an experiment."""
    experiment, filename, run_or_load = args
    if run_or_load == "load":
        if os.path.isfile(filename):
            with open(filename, "rb") as f:
                results = pickle.load(f)
        else:
            run_or_load = "run"
    if run_or_load == "run":
        results = experiment_f(experiment)
        with open(filename, "wb") as f:
            pickle.dump(results, f, protocol=-1)
    return results


def run_experiments_parameters(
    experiment,
    run_or_load,
    results_dir,
    parameters={"reps": 1},
    num_processes=8,
    experiment_f=run_single_experiment,
):
    """Create results for Exqaliber AE for different parameters."""
    # create the folder structure
    param_bytes = pickle.dumps(parameters)

    # Generate a hash value based on the serialized parameter dictionary
    hash_obj = hashlib.sha256(param_bytes)
    hash_value = hash_obj.hexdigest()

    # Use the hash value as the folder name
    folder_name = f"experiment_{hash_value[:8]}"
    results_dir = os.path.join(results_dir, folder_name)

    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

        # Save the parameter dictionary as a .pkl file inside the folder
        params_filename = "params.pkl"
        params_path = os.path.join(results_dir, params_filename)

        with open(params_path, "wb") as f:
            pickle.dump(parameters, f)

    # recording the results
    results = {}
    results["parameters"] = parameters.copy()

    # load the parameters
    experiment["max_iter"] = parameters.get("max_iter", 0)

    iterables = []
    fixed = []
    for key, value in parameters.items():
        if isinstance(value, (list, tuple, set, np.ndarray)):
            iterables.append(key)
        else:
            fixed.append(key)
    results["iterables"] = iterables.copy()
    results["fixed"] = fixed.copy()

    for fixed_value in fixed:
        experiment[fixed_value] = parameters[fixed_value]

    n = parameters.get("reps", 1)

    # create queues
    qs = [Queue()]
    q_lens = [0]
    q_idx = 0
    num_jobs = 0

    for values in product(*[parameters[key] for key in iterables]):
        # setting the experiment
        for iterable, value in zip(iterables, values):
            experiment[iterable] = value

        # creating the experiment in the queue for reps repetitions.
        for i in range(n):
            filename = os.path.join(results_dir, f"{num_jobs:06d}.pkl")
            try:
                qs[q_idx].put(
                    (experiment.copy(), filename, run_or_load),
                    block=True,
                    timeout=0.1,
                )
                q_lens[q_idx] += 1
            except queue.Full:
                qs.append(Queue())
                q_lens.append(0)
                q_idx += 1
                qs[q_idx].put(
                    (experiment.copy(), filename, run_or_load),
                    block=True,
                    timeout=0.1,
                )
                q_lens[q_idx] += 1
            num_jobs += 1

    pool = Pool(np.min([num_jobs, num_processes]))
    output = []

    for q_len, q in tqdm(zip(q_lens, qs), total=len(qs)):
        with tqdm(
            total=q_len, position=1, leave=False
        ) as pbar:  # create a progress bar with the total number of jobs
            imap_func = partial(
                run_experiment_single_rep, experiment_f=experiment_f
            )
            imap_iter = pool.imap(imap_func, [q.get() for _ in range(q_len)])
            for result in imap_iter:
                output.append(result)
                pbar.update(1)

    pool.close()
    pool.join()

    for values in product(*[parameters[key] for key in iterables]):
        n = parameters.get("reps", 1)
        for i in range(n):
            filename = os.path.join(results_dir, f"{num_jobs:06d}.pkl")
            # q.put((experiment.copy(), filename, run_or_load))

            results[(*values, i)] = output.pop(0)

    return results


def get_results_slice(results, rules={}):
    """Get results sliced based on a rule."""
    out = {}

    index = {}
    for k, v in rules.items():
        if k in results["iterables"]:
            index[results["iterables"].index(k)] = v
        if k in results["fixed"]:
            continue

    if len(index) == 0:
        for k, v in results.items():
            if isinstance(k, str):
                continue
            out[k] = v

    for k, v in results.items():
        if isinstance(k, str):
            continue
        if [k[i] for i in index] == list(index.values()):
            out[k] = v

    return out


if __name__ == "__main__":
    # saving and running parameters
    run_or_load = "load"
    save_results = True
    show_results = False
    one_theta_experiment = False
    # sweep_experiment = True

    # parameters all experiments
    epsilon_target = 1e-3
    # prior_mean = np.pi/2
    prior_mean = "true_theta"
    prior_std = 1
    method = "greedy"
    EXPERIMENT = {
        "epsilon_target": epsilon_target,
        "prior_mean": prior_mean,
        "prior_std": prior_std,
        "method": method,
    }

    # parameters one run experiment
    true_theta = 0.00416
    do_animation_plot = False
    do_convergence_plot = True

    # parameters theta sweep
    reps = 5000
    resolution = 180
    theta_range = np.linspace(0, np.pi, resolution, endpoint=True)
    # replace theta == 0.0 with 2pi
    theta_range[0] = 2 * np.pi
    do_circular_histogram = False
    do_accuracy_plot_linear = False
    do_error_plot = True

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

    # if sweep_experiment:
    #     results_dir = f"results/{resolution}x{reps}"

    #     results_multiple_thetas = run_experiment_multiple_thetas(
    #         theta_range,
    #         experiment=EXPERIMENT,
    #         run_or_load=run_or_load,
    #         results_dir=results_dir,
    #         reps=reps,
    #     )

    #     if do_circular_histogram:
    #         filename = (
    #             f"{results_dir}/figures/circular_histogram.png"
    #             if save_results
    #             else False
    #         )
    #         circular_histogram(
    #             results_multiple_thetas,
    #             theta_range,
    #             experiment=EXPERIMENT,
    #             save=filename,
    #             show=show_results,
    #         )

    #     if do_accuracy_plot_linear:
    #         filename = (
    #             f"{results_dir}/accuracy_linear.png"
    #             if save_results else False
    #         )
    #         accuracy_plot_linear(
    #             results_multiple_thetas,
    #             theta_range,
    #             experiment=EXPERIMENT,
    #             save=filename,
    #             show=show_results,
    #         )

    #     if do_error_plot:
    #         filename = (
    #             f"{results_dir}/figures/error_in_estimate-1.png"
    #             if save_results
    #             else False
    #         )
    #         error_in_estimate_2d_hist(
    #             results_multiple_thetas,
    #             theta_range,
    #             experiment=EXPERIMENT,
    #             save=filename,
    #             show=show_results,
    #         )

    print("Done.")
