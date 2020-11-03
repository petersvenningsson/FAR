###########
# IMPORTS #
###########
# 3rd-party
import numpy as np
from scipy import stats

###########
# CLASSES #
###########


class Distribution:
    def __init__(self, obj, **kwargs):
        self.distribution_object = obj(**kwargs)
        self.params = kwargs
        self.dimensionality = len(self.draw())

    def draw(self):
        return self.distribution_object.rvs()

    def likelihood(self, x):
        assert len(x) == self.dimensionality
        return self.distribution_object.pdf(x)


###########
# GLOBALS #
###########
mean_A = [0.75, -0.25]
mean_B = [1, 0.25]
mean_C = [-0.5, 2]
cov_A = [[1.25 / 2, 0], [0, 1.25 / 2]]
cov_B = [[1.25 / 2, 0], [0, 1.25 / 2]]
cov_C = [[1, 0.5], [0.5, 1.25]]

class_A = Distribution(stats.multivariate_normal, **{"mean": mean_A, "cov": cov_A})
class_B = Distribution(stats.multivariate_normal, **{"mean": mean_B, "cov": cov_B})
class_C = Distribution(stats.multivariate_normal, **{"mean": mean_C, "cov": cov_C})

##########
# SCRIPT #
##########

if __name__ == "__main__":
    dist = stats.multivariate_normal

    arg = {"loc": 1, "scale": 2}

    mydist = Distribution(dist, **arg)
    print(mydist.draw())
