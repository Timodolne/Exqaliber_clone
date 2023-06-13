"""Graph the behaviour of Exqaliber Amplitude Estimation."""
import hashlib
import os.path
import pickle
import queue
import warnings
from fractions import Fraction
from functools import partial
from itertools import product
from multiprocessing import Pool, Queue

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
from qiskit.algorithms.amplitude_estimators.amplitude_estimator import (
    AmplitudeEstimatorResult,
)
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
    if np.abs(x) < 1e-8:
        return f"{x:{format_string}}"
    frac = Fraction(x / np.pi).limit_denominator(max_denominator)

    # check if the fraction matches
    if np.abs(x - frac * np.pi) < 1e-8:
        # whole fraction
        if frac.numerator == 1:
            if frac.denominator == 1:
                return r"$\pi$"
            return rf"$\pi/{frac.denominator}$"
        elif frac.numerator == -1:
            if frac.denominator == 1:
                return r"$-\pi$"
            return rf"$-\pi/{frac.denominator}$"
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
    y = [np.log(dist.standard_deviation) for dist in distributions]
    x = range(len(y))

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
    x = range(len(y))
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
    results: dict,
    save: bool = False,
    show: bool = True,
    rules: dict = None,
    experiment: dict = None,
    val: float = np.pi / 2,
):
    """Plot the error in the estimate."""
    if isinstance(
        results["parameters"]["epsilon_target"],
        (list, tuple, set, np.ndarray),
    ):
        if "epsilon_target" in rules.keys():
            epsilon_target = rules["epsilon_target"]
        else:
            raise "Choose one epsilon target for circular histogram."

    if rules is None:
        rules = {"zeta": 0}
    # get queries
    results_sliced = get_results_slice(results, rules=rules)

    thetas = np.array([res.true_theta for res in results_sliced.values()])
    estimations = np.array(
        [res.final_theta for res in results_sliced.values()]
    )

    theta_range = results["parameters"]["true_theta"]

    nb_reps = results["parameters"]["reps"]

    errors = estimations - thetas
    # wrapping around the circle
    thetas = np.mod(thetas, 2 * np.pi)
    theta_range = np.mod(theta_range, 2 * np.pi)
    errors = np.mod(errors + np.pi, 2 * np.pi) - np.pi

    # data
    width = np.max(theta_range[1:]) / len(theta_range)
    min_y = errors.min()
    max_y = errors.max()

    # build figure
    fig, axs = plt.subplots(2, 2, figsize=(12, 9))

    # Left figures
    # top figure
    bins_top = np.linspace(-3 * epsilon_target, 3 * epsilon_target, 100)
    i = np.argwhere(theta_range == val)

    axs[0, 0].hist(errors.reshape(-1, nb_reps)[i].flatten(), bins=bins_top)

    # bottom figure
    bins_bottom = np.linspace(min_y, max_y, 100)
    axs[1, 0].hist(errors.reshape(-1, nb_reps)[i].flatten(), bins=bins_bottom)

    # Right figures
    # top figure
    x_bins = np.linspace(
        -width / 2, np.pi + width / 2, len(theta_range) + 1, endpoint=True
    )
    y_bins = bins_top
    bins = [x_bins, y_bins]

    axs[0, 1].hist2d(
        thetas,
        errors,
        bins=bins,
        cmin=1,
        cmax=nb_reps,
        norm=mpl.colors.LogNorm(),
    )
    axs[0, 1].axvline(val, ymin=-1, ymax=1, linestyle="--", color="red")

    # 1-alpha conf interval
    alpha = experiment["alpha"]
    label = rf"{100*(1-alpha):.0f}% conf. interval"
    errors_reshaped = errors.reshape(-1, nb_reps)
    q_upp = np.quantile(errors_reshaped, (1 - alpha / 2), axis=1)
    q_down = np.quantile(errors_reshaped, alpha / 2, axis=1)

    axs[0, 1].fill_between(
        theta_range, q_upp, q_down, alpha=0.4, color="red", label=label
    )
    axs[0, 1].legend()

    # bottom figure
    x_bins = np.linspace(
        -width / 2, np.pi + width / 2, len(theta_range) + 1, endpoint=True
    )
    y_bins = bins_bottom
    bins = [x_bins, y_bins]

    h2 = axs[1, 1].hist2d(
        thetas,
        errors,
        bins=bins,
        cmin=1,
        cmax=nb_reps,
        norm=mpl.colors.LogNorm(),
    )
    axs[1, 1].axvline(val, ymin=min_y, ymax=max_y, linestyle="--", color="red")

    axs[1, 1].fill_between(
        theta_range, q_upp, q_down, alpha=0.4, color="red", label=label
    )
    axs[1, 1].legend()

    # X-axis right
    for ax in axs[:, 1]:
        ax.set_xlabel(r"$\theta$")
        ax.set_xlim(-width / 2, np.pi + width / 2)

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

        ax.set_ylim(0.5, None)
        ax.set_yscale("log")
        ax.set_ylabel("Frequency")

    axs[0, 0].set_title(rf"Errors in estimates for $\theta=${val:.3f}")

    # Finishing figure
    experiment_epsilon = experiment.copy()
    experiment_epsilon["epsilon_target"] = epsilon_target
    title = (
        f"Error in estimate of theta "
        f"\n{experiment_string(experiment_epsilon, True)}"
    )
    plt.suptitle(title)

    # Colorbar
    # make room for colorbar and slider
    fig.subplots_adjust(bottom=0.1, right=0.95)
    fig.colorbar(h2[3], ax=axs[:, 1], label="Runs")

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


# TODO Does this need a log warning?
def run_experiment_single_rep(args, experiment_f=run_single_experiment):
    """Run a single repetition of an experiment."""
    experiment, filename, run_or_load = args

    if run_or_load == "load" and os.path.isfile(filename):
        with open(filename, "rb") as f:
            results = pickle.load(f)
        return results

    results = experiment_f(experiment)
    with open(filename, "wb") as f:
        pickle.dump(results, f, protocol=-1)
    return results


# TODO update the variable names here
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
            experiment[key] = value

    results["iterables"] = iterables.copy()
    results["fixed"] = fixed.copy()

    n = parameters.get("reps", 1)
    num_jobs = 0

    # create queues
    qs = [Queue()]
    q_lens = [0]

    for i_parameter_settings in product(
        *[parameters[key] for key in iterables]
    ):
        # setting the experiment
        for j_parameter_name, j_parameter_value in zip(
            iterables, i_parameter_settings
        ):
            experiment[j_parameter_name] = j_parameter_value

        # creating the experiment in the queue for reps repetitions.
        for j_rep in range(n):
            filename = os.path.join(results_dir, f"{num_jobs:06d}.pkl")

            try:
                qs[-1].put(
                    (experiment.copy(), filename, run_or_load),
                    block=True,
                    timeout=0.1,
                )
            except queue.Full:
                qs.append(Queue())
                q_lens.append(0)
                qs[-1].put(
                    (experiment.copy(), filename, run_or_load),
                    block=True,
                    timeout=0.1,
                )

            q_lens[-1] += 1
            num_jobs += 1

    with Pool(np.min([num_jobs, num_processes])) as pool:
        experiment_results = run_pooled_experiments(
            pool, experiment_f, qs, q_lens
        )
    pool.join()

    for values in product(*[parameters[key] for key in iterables]):
        n = parameters.get("reps", 1)
        for i_rep in range(n):
            results[(*values, i_rep)] = experiment_results.pop(0)

    return results


def task_generator(queue: Queue) -> tuple[dict, str, str]:
    """Convert Queue to iterable.

    Parameters
    ----------
    queue : Queue
        Queue object containing sets of parameters, the name of the file
        to save to and whether to try and load from file first or run
        the algorithm regardless.

    Yields
    ------
    tuple[dict, str, str]
        An experiment instance, containing a parameter setting, the file
        name to save the experiment to and whether to try and load from
        file first or run the algorithm.
    """
    while not queue.empty():
        yield queue.get()


def run_pooled_experiments(
    pool: Pool,
    experiment_evaluation_function: callable,
    experiment_queues: list[Queue],
    experiment_queue_lengths: list[int],
) -> list[AmplitudeEstimatorResult]:
    """Run experiments using multiple CPUs.

    Parameters
    ----------
    pool : Pool
        Pool of CPUs to assign processes to.
    experiment_evaluation_function : callable
        The algorithm to call in each process.
    experiment_queues : list[Queue]
        Experiment setups as a series of queues.
    experiment_queue_lengths : list[int]
        The length of each queue for the progress bar.

    Returns
    -------
    list[AmplitudeEstimatorResult]
        Results from the experiments.
    """
    experiment_results = []
    imap_func = partial(
        run_experiment_single_rep, experiment_f=experiment_evaluation_function
    )
    for i_queue_length, i_queue in tqdm(
        zip(experiment_queue_lengths, experiment_queues),
        total=len(experiment_queues),
    ):
        with tqdm(
            total=i_queue_length, position=1, leave=False
        ) as pbar:  # create a progress bar with the total number of jobs

            imap_iter = pool.imap(imap_func, task_generator(i_queue))

            for result in imap_iter:
                experiment_results.append(result)
                pbar.update(1)

    return experiment_results


def get_results_slice(
    results: dict,
    rules: dict[str, float | int | list | tuple | set | np.ndarray] = None,
) -> dict[tuple[float], AmplitudeEstimatorResult]:
    """Get results sliced based on a rule.

    Parameters
    ----------
    results : dict
        Results from `run_experiments_parameters`. Should contain
        keys corresponding to 'fixed', 'iterables', 'parameters' and
        tuples with the chosen iterable parameters.
    rules : dict, optional
        Pairs of parameter names and allowed values for those
        parameters, by default None.

    Returns
    -------
    dict[tuple[float], AmplitudeEstimatorResult]
        Pairs of the iterable parameter values and the corresponding
        algorithm result.
    """
    rules = rules or {}
    out = {}

    iterables = results.get("iterables", [])
    fixed = results.get("fixed", [])

    # Convert all rules to iterables to simplify checks
    rules = {
        k: (v if isinstance(v, (list, tuple, set, np.ndarray)) else [v])
        for k, v in rules.items()
    }

    # If any fixed parameters don't match rules, return an empty dict
    for i_parameter_name in fixed:

        # Skip checks where the parameter isn't specified by a rule.
        if i_parameter_name not in rules:
            continue

        i_fixed_value = results.get("parameters").get(i_parameter_name)
        i_allowed_values = rules.get(i_parameter_name)

        if i_fixed_value not in i_allowed_values:
            warnings.warn(
                f"Fixed parameter {i_parameter_name}={i_fixed_value} doesn't "
                f"match allowed values {i_allowed_values}, returning an empty "
                "dictionary."
            )
            return {}

    # Create a dictionary mapping iterable parameters to their
    # corresponding indices.
    indices = {iterables.index(k): rules[k] for k in rules if k in iterables}

    # Iterate over the results
    for k, v in results.items():
        # Only get experiment results, not other keys
        if isinstance(k, str):
            continue

        # If the value at each indexed position matches the rule, add it
        # to the output
        if all(k[i] in indices.get(i, [k[i]]) for i in range(len(k))):
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
