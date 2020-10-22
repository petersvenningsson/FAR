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

###########
# GLOBALS #
###########
cls_prior = np.array([1/3,1/3,1/3])
distributions = [class_A, class_B, class_C]
distributions_colors = ['g', 'y', 'r']

# Visualization params
draw_speed = 0.001
#############
# FUNCTIONS #
#############

def calculate_posterior(prior, likelihoods):
    total_likelihood = (prior*likelihoods).sum()
    posterior = prior*likelihoods / total_likelihood
    return posterior

##########
# SCRIPT #
##########
sample_history = []
likelihood_history = []

belief = cls_prior

fig = plt.figure()
while True:
    # Static artist
    ax1 = fig.add_subplot(121)
    for c, color in zip(distributions, distributions_colors):
        confidence_ellipse(np.array(c.params['mean']), np.array(c.params['cov']), ax1, n_std = 2.0, facecolor='none', edgecolor=color, linewidth=1.5)
    
    ## Dynamic artist
    # Scatter
    if sample_history:
        _samples = np.array(sample_history)
        sns.scatterplot(x=_samples[:,0], y=_samples[:,1], size = likelihood_history, ax = ax1, legend = 0)

    # Histogram
    ax2 = fig.add_subplot(122)
    ax2.set(ylim=(0, 1))
    sns.barplot(x=["Correct","Proximal","Distant"], y = belief, ax = ax2, palette = distributions_colors)

    # Draw
    ax1.axis('equal')
    fig.canvas.draw_idle()
    plt.pause(draw_speed)
    fig.clf()

    # Generate time step
    sample = class_A.draw()
    likelihood = sum([c.likelihood(sample) for c in distributions])
    likelihood_history.append(likelihood)
    sample_history.append(sample)

    # Bayesian update
    belief = calculate_posterior(prior = belief, likelihoods = np.array([ c.likelihood(sample) for c in distributions ]))