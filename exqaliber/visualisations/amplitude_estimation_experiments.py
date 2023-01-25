"""Graph the behaviour of Exqaliber Amplitude Estimation."""
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
from scipy.stats import norm

from exqaliber.exqaliber_amplitude_estimation import (
    ExqaliberAmplitudeEstimation,
)

if __name__ == "__main__":
    EXPERIMENT = {
        "true_theta": 0.3,
        "prior_mean": 0.32,
        "prior_variance": 0.03,
        "method": "greedy",
    }

    ae = ExqaliberAmplitudeEstimation(0.01, 0.01, **EXPERIMENT)
    estimation_problem = None

    result = ae.estimate(estimation_problem)
    distributions = result.distributions
    n_iter = len(distributions)

    print(f"Executed {len(result.powers)} rounds")
    print(
        f"Finished with variance of {result.variance:.6f} "
        f"and mean {result.estimation:.6f}, "
        f"(true theta: {EXPERIMENT['true_theta']})."
    )

    if n_iter < 1000:
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

        axs[0].set_xlim(0, 1)
        axs[1].set_xlim(0, 1)
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
                    axs[0].yaxis.get_major_ticks()[i].tick1line.set_visible(
                        False
                    )
                    axs[0].yaxis.get_major_ticks()[i].tick2line.set_visible(
                        False
                    )
                else:
                    axs[0].yaxis.get_major_ticks()[i].label1.set_visible(True)
                    axs[0].yaxis.get_major_ticks()[i].tick1line.set_visible(
                        True
                    )
                    axs[0].yaxis.get_major_ticks()[i].tick2line.set_visible(
                        True
                    )

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

        anim.save("animation.mp4", fps=30, extra_args=["-vcodec", "libx264"])

        # plt.show()

    fig, axs = plt.subplots(1, 2, figsize=(10, 5))

    # Variance plot
    x = range(n_iter)
    y = [dist.standard_deviation for dist in distributions]
    axs[0].plot(x, y)

    axs[0].set_xlim(0, n_iter)
    axs[0].set_ylim(0, max(y) * 1.05)

    axs[0].set_xlabel("Iteration")
    axs[0].set_ylabel(r"$\sigma^2$")

    # Mean plot
    y = [dist.mean for dist in distributions]
    axs[1].plot(x, y, label=r"$\mu$")
    axs[1].hlines(
        y=EXPERIMENT["prior_mean"],
        xmin=0,
        xmax=n_iter,
        label=r"Prior $\theta$",
        linestyles="dotted",
    )
    axs[1].hlines(
        y=EXPERIMENT["true_theta"],
        xmin=0,
        xmax=n_iter,
        label=r"True $\theta$",
        linestyles="--",
    )

    axs[1].set_xlim(0, n_iter)
    axs[1].set_ylim(0, 1)

    axs[1].set_xlabel("Iteration")
    axs[1].set_ylabel(r"$\mu$")

    axs[1].legend()

    plt.show()

    print("Done.")
