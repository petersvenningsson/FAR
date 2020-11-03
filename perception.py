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
draw_speed = 0.5

###########
# CLASSES #
###########

class PerceptualProcessor():
    def __init__(self, transition_matrix = None):
        self.transition_matrix = transition_matrix

    def calculate_posterior(self, sample, prior, distributions):
        if sample is None:
            return prior

        likelihoods = np.array([ c.likelihood(sample) for c in distributions ])
        total_likelihood = (prior*likelihoods).sum()
        posterior = prior*likelihoods / total_likelihood
        return posterior
    
    def transition(self, prior):
        assert self.transition_matrix is not None, "perceptual processor without transition matrix"
        posterior = self.transition_matrix @ prior
        return posterior

