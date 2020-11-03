###########
# IMPORTS #
###########
# 3rd-party
from execution import ExecutiveProcessor
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

sns.set_theme(style="darkgrid")

# local
from distributions import class_A, class_B, class_C
from perception import PerceptualProcessor
from execution import ExecutiveProcessor
from utils import confidence_ellipse, Collector

###########
# GLOBALS #
###########
CLS_PRIOR = np.array([1 / 3, 1 / 3, 1 / 3])
TRANSITION_MATRIX = np.array(
    [[0.99, 0.005, 0.005], [0.005, 0.99, 0.005], [0.005, 0.005, 0.99]]
)
ENTROPY_THRESHOLD = 0.3

distributions = [class_A, class_B, class_C]
distributions_colors = ["g", "y", "r"]
# Visualization params
draw_speed = 0.1

#############
# FUNCTIONS #
#############


def render(fig, belief, collector):
    # Static artist
    ax1 = fig.add_subplot(221)
    ax1.axes.set_title("Feature distributions")
    for c, color in zip(distributions, distributions_colors):
        confidence_ellipse(
            np.array(c.params["mean"]),
            np.array(c.params["cov"]),
            ax1,
            n_std=2.0,
            facecolor="none",
            edgecolor=color,
            linewidth=1.5,
        )

    ## Dynamic artist
    # Scatter
    if collector.sample_history is not None:
        _samples = np.array(collector.sample_history)
        sns.scatterplot(
            x=_samples[:, 0],
            y=_samples[:, 1],
            size=collector.likelihood_history,
            ax=ax1,
            legend=0,
        )

    # Histogram
    if belief is not None:
        ax2 = fig.add_subplot(222)
        ax2.set(ylim=(0, 1))
        ax2.axes.set_ylabel("Class probability")
        sns.barplot(
            x=["Correct", "Proximal", "Distant"],
            y=belief,
            ax=ax2,
            palette=distributions_colors,
        )

    ax3 = fig.add_subplot(223)
    ax3.set(ylim=(0, 1.5))
    labels = []
    if collector.perception_entropy_history:
        sns.lineplot(
            x=range(len(collector.perception_entropy_history)),
            y=collector.perception_entropy_history,
            dashes=True,
            ax=ax3,
            marker="o",
        )
        labels.append("Perception entropy")

    if collector.execution_entropy_history:
        sns.lineplot(
            x=range(len(collector.execution_entropy_history)),
            y=collector.execution_entropy_history,
            ax=ax3,
            marker="o",
        )
        labels.append("Execution entropy")

    if collector.action_history:
        actions = [
            s
            for s, b in zip(
                range(len(collector.action_history)), collector.action_history
            )
            if not b
        ]
        sns.scatterplot(x=actions, y=[1.3 for _ in range(len(actions))], marker="X")
        labels.append("Skipped measurements")

    plt.legend(labels=labels, loc = 2)

    # Draw
    ax1.axis("equal")
    fig.canvas.draw_idle()
    plt.pause(draw_speed)
    fig.clf()


def main():
    # Initialization
    collector = Collector(
        sample_history=[],
        likelihood_history=[],
        perception_entropy_history=[],
        execution_entropy_history=[],
        action_history=[],
        t=0,
    )
    belief = CLS_PRIOR

    # Instantiation
    perceptual = PerceptualProcessor(transition_matrix=TRANSITION_MATRIX)
    execution = ExecutiveProcessor(entropy_threshold=ENTROPY_THRESHOLD)
    execution_entropy = 1
    fig = plt.figure()

    while True:
        # Generate time step
        collector.action_history.append(execution.take_action(execution_entropy))

        if execution.take_action(execution_entropy):
            sample = class_A.draw()
        else:
            sample = None

        if sample is not None:
            likelihood = sum([c.likelihood(sample) for c in distributions])
            collector.likelihood_history.append(likelihood)
            collector.sample_history.append(sample)

        # Bayesian update
        belief = perceptual.calculate_posterior(sample, belief, distributions)
        collector.perception_entropy_history.append(stats.entropy(belief, base=2))
        render(fig, belief, collector)

        collector.t += 1

        # Calculate transition
        belief = perceptual.transition(prior=belief)
        execution_entropy = stats.entropy(belief, base=2)
        collector.execution_entropy_history.append(execution_entropy)
        render(fig, belief, collector)


##########
# SCRIPT #
##########
if __name__ == "__main__":
    main()
