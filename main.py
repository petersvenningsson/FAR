###########
# IMPORTS #
###########
# 3rd-party
from execution import ExecutiveProcessor
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from utils import confidence_ellipse
sns.set_theme(style="darkgrid")
# local
from distributions import class_A, class_B, class_C
from perception import PerceptualProcessor
from execution import ExecutiveProcessor
from scipy import stats
###########
# GLOBALS #
###########
CLS_PRIOR = np.array( [1/3, 1/3, 1/3] )
TRANSITION_MATRIX = np.array( [[0.99, 0.005, 0.005], [0.005, 0.99, 0.005], [0.005, 0.005, 0.99]] )
ENTROPY_THRESHOLD = 0.3

distributions = [class_A, class_B, class_C]
distributions_colors = ['g', 'y', 'r']
# Visualization params
draw_speed = 0.1

#############
# FUNCTIONS #
#############

def render(fig, t = None, belief = None, sample_history = [], likelihood_history = [], perception_entropy_history = [], execution_entropy_history = [], action_history = []):
    # Static artist
    ax1 = fig.add_subplot(221)
    ax1.axes.set_title('Feature distributions')
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
        ax2.axes.set_ylabel('Class probability')
        sns.barplot(x=["Correct","Proximal","Distant"], y = belief, ax = ax2, palette = distributions_colors)

    ax3 = fig.add_subplot(223)
    ax3.set(ylim=(0, 1.5))
    if perception_entropy_history:
        sns.lineplot(x = range(len(perception_entropy_history)), y = perception_entropy_history,  dashes=True, ax = ax3, marker='o')
    
    if execution_entropy_history:    
        sns.lineplot(x = range(len(execution_entropy_history)), y = execution_entropy_history, ax = ax3, marker="o")
    
    if action_history:
        actions = [s for s,b in zip(range(len(action_history)), action_history) if not b]
        sns.scatterplot(x = actions, y = [1.3 for _ in range(len(actions))], marker = 'X')

    plt.legend(labels=['Perception entropy', 'Execution entropy', 'Skipped measurements'])
        

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
    action_history = []

    t = 0
    belief = CLS_PRIOR

    # Instantiation
    perceptual = PerceptualProcessor(transition_matrix = TRANSITION_MATRIX )
    execution = ExecutiveProcessor(entropy_threshold = ENTROPY_THRESHOLD)
    execution_entropy = 1
    fig = plt.figure()

    while True:
        # Generate time step
        action_history.append(execution.take_action(execution_entropy))

        if execution.take_action(execution_entropy):
            sample = class_A.draw()
        else:
            sample = None

        if sample is not None:
            likelihood = sum([c.likelihood(sample) for c in distributions])
            likelihood_history.append(likelihood)
            sample_history.append(sample)

        # Bayesian update
        belief = perceptual.calculate_posterior(sample, belief, distributions)
        perception_entropy_history.append( stats.entropy(belief, base = 2))
        render(fig, t, belief, sample_history, likelihood_history, perception_entropy_history, execution_entropy_history, action_history)

        t += 1

        # Calculate transition
        belief = perceptual.transition( prior = belief )
        execution_entropy = stats.entropy( belief, base = 2 )
        execution_entropy_history.append( execution_entropy )
        render(fig, t, belief, sample_history, likelihood_history, perception_entropy_history, execution_entropy_history, action_history)


##########
# SCRIPT #
##########
if __name__ == '__main__':
    main()