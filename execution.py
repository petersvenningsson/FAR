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

class ExecutiveProcessor():
    def __init__(self, entropy_threshold):
        self.entropy_threshold = entropy_threshold

    def take_action(self, entropy):
        if entropy > self.entropy_threshold:
            return True
        else:
            return False