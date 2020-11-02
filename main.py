###########
# IMPORTS #
###########
# 3rd-party
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from utils import confidence_ellipse
sns.set_theme(style="darkgrid")
# local
from distributions import class_A, class_B, class_C
from perception import PerceptualProcessor
from scipy import stats
###########
# GLOBALS #
###########
CLS_PRIOR = np.array( [1/3, 1/3, 1/3] )
TRANSITION_MATRIX = np.array( [[0.99, 0.005, 0.005], [0.005, 0.99, 0.005], [0.005, 0.005, 0.99]] )

distributions = [class_A, class_B, class_C]
distributions_colors = ['g', 'y', 'r']
# Visualization params
draw_speed = 0.5

#############
# FUNCTIONS #
#############

def render(fig, t = None, belief = None, sample_history = [], likelihood_history = [], perception_entropy_history = [], execution_entropy_history = []):
    # Static artist
    ax1 = fig.add_subplot(221)
    for c, color in zip(distributions, distributions_colors):
        confidence_ellipse(np.array(c.params['mean']), np.array(c.params['cov']), ax1, n_std = 2.0, facecolor='none', edgecolor=color, linewidth=1.5)

    ## Dynamic artist
    # Scatter
    if sample_history is not None:
        _samples = np.array(sample_history)
        sns.scatterplot(x=_samples[:,0], y=_samples[:,1], size = likelihood_history, ax = ax1, legend = 0)

    # Histogram
    if belief is not None:
        ax2 = fig.add_subplot(222)
        ax2.set(ylim=(0, 1))
        sns.barplot(x=["Correct","Proximal","Distant"], y = belief, ax = ax2, palette = distributions_colors)

    if perception_entropy_history and perception_entropy_history:
        ax3 = fig.add_subplot(223)
        sns.lineplot(x = range(len(perception_entropy_history)), y = perception_entropy_history,  dashes=True, ax = ax3, marker='o')
        sns.lineplot(x = range(len(execution_entropy_history)), y = execution_entropy_history, ax = ax3, marker="o")
        ax3.set(ylim=(0, 1.5))
        plt.legend(labels=['Perception entropy', 'Execution entropy'])


    # Draw
    ax1.axis('equal')
    fig.canvas.draw_idle()
    plt.pause(draw_speed)
    fig.clf()

def main():
    # Initialization
    sample_history = []
    likelihood_history = []
    perception_entropy_history = []
    execution_entropy_history = []

    t = 0
    belief = CLS_PRIOR

    # Instantiation
    perceptual = PerceptualProcessor(transition_matrix = TRANSITION_MATRIX )
    fig = plt.figure()
    while True:

        # Generate time step
        sample = class_A.draw()
        likelihood = sum([c.likelihood(sample) for c in distributions])
        likelihood_history.append(likelihood)
        sample_history.append(sample)

        # Bayesian update
        belief = perceptual.calculate_posterior(sample, belief, distributions)
        perception_entropy_history.append( stats.entropy(belief, base = 2))
        render(fig, t, belief, sample_history, likelihood_history, perception_entropy_history, execution_entropy_history)

        t += 1

        # Calculate transition
        belief = perceptual.transition(prior = belief)
        execution_entropy_history.append( stats.entropy(belief, base = 2))
        render(fig, t, belief, sample_history, likelihood_history, perception_entropy_history, execution_entropy_history)


##########
# SCRIPT #
##########
if __name__ == '__main__':
    main()