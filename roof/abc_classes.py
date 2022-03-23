import numpy as np
from scipy.stats import loguniform

class lambda_prior:
    """
    This example class defines an example prior class to handle both evaluations and 
    sampling of the prior. This samples values for the slope of pixels in an image. 
    """

    def sample(self, nsamples=None):
        """
        Function that sample points from the prior. Uniform for the slope and intercept --- log-uniform for the standard-deviation.
        """
        if nsamples is None:
            nsamples = self.nsamples

        # Evaluate samples:
        lambda_samples = np.random.uniform(self.l1, self.l2, nsamples*self.ngroups)

        # Return them:
        return lambda_samples

    def validate(self, theta):
        """
        This function validates that the set of parameters to evaluate 
        are within the ranges of the prior
        """

        # Extract current parameters to evaluate the priors on:
        lamb = theta

        # Validate the uniform priors:
        if lamb <= self.l1 or lamb >= self.l2:
            return False

        # If all is good, return a nice True:
        return True

    def evaluate(self, theta):
        """
        Given an input vector, evaluate the prior. In this case, this just returns the 
        priors defined by the hyperparameters. For the uniform prior, the value of the 
        prior doesn't depend on the inputs. For the loguniform, that's note the case.
        """

        # Return the prior evaluated on theta:
        return self.lamb_prior


    def __init__(self, l1 = 0, l2 = 5000, nsamples = 9, ngroups=5):

        # Define hyperparameters of the prior. First for slope (a, uniform prior):
        self.l1 = l1
        self.l2 = l2

        # Value of the prior given hyperparameters:
        self.lamb_prior = 1. / (l2 - l1)

        # Define the default number of samples:
        self.nsamples = nsamples

        # Define the number of groups
        self.ngroups = ngroups

class example_simulator:
    """
    This example class generates a simulator object that is able to simulate several or 
    single simulations. Simulates same data as the one in the `gen_fake_data()` function.
    """

    def single_simulation(self, parameters):

        # Extract parameters:
        a, b, sigma = parameters

        return a * self.x + b + np.random.normal(0, sigma, self.length)

    def several_simulations(self, parameters):

        # Extract parameters:
        all_as, all_bs, all_sigmas = parameters
        nsamples = len(all_sigmas)

        # Define array to store simulations:
        simulations = np.zeros([nsamples, self.length])

        # Lazy loop to do several of these; could apply multi-processing:
        for i in range(nsamples):
            simulations[i,:] = self.single_simulation([all_as[i], all_bs[i], all_sigmas[i]])

        return simulations

    def __init__(self,ngroups = 5):
        self.ngroups = ngroups
        self.x = np.arange(self.ngroups)

class example_distance:
    """
    Example class for distance.
    """
    def slopes_distance(self,simulation):
        """ Given a dataset and a simulation, this function returns the distance 
            between them. This is defined here as the least square distance between the
            simulations and data points """ 

        sim_slopes = [np.polyfit(self.xdata,ramp,1)[0] for ramp in simulation]
        return np.sum([np.abs(self.yslopes[i] - sim_slopes[i]) / self.data_slope[i] for i in range(self.ngroups)])

    def counts_distance(self,simulation):
        """ Given a dataset and a simulation, this function returns the distance 
            between them. This is defined here as the distance between the simulated and data points"""

        return np.sum([((self.ydata[i] - simulation[i])/self.data[i])**2 for i in range(self.ngroups)]) 


    def several_distances(self, simulations):
        """ Same as single distance, several times """

        nsimulations = simulations.shape[0]
        distances = np.zeros(nsimulations)
        for i in range(nsimulations):
            if self.distance_fnc = 'slopes':
                distances[i] = self.slopes_distance(simulation)
            elif self.distance_fnc = 'counts':
                distances[i] = self.counts_distance(simulation)
        return distances

    def __init__(self, ydata,distance_fn):

        self.xdata = np.linspace(-5, 5, length)
        self.ydata = ydata
        self.data_slopes = [np.polyfit(self.xdata,ramp,1)[0] for ramp in simulation]

        self.distance_fn = distance_fn