import numpy as np
from scipy.stats import loguniform

from stochastic.processes.noise import ColoredNoise

from . import utils

class slope_prior:
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
        slope_samples = [np.random.uniform(self.s1, self.s2, nsamples) for i in range(9)]

        # Return them:
        return slope_samples

    def validate(self, theta):
        """
        This function validates that the set of parameters to evaluate 
        are within the ranges of the prior
        """

        # Extract current parameters to evaluate the priors on:
        slopes = theta

        # Validate the uniform priors:
        for slope in slopes:
            if slope <= self.s1 or slope >= self.s2:
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
        return self.slope_prior * 9


    def __init__(self, s1=0, s2=5000, nsamples=9):

        # Define hyperparameters of the prior. First for slope (a, uniform prior):
        self.s1 = s1
        self.s2 = s2

        # Value of the prior given hyperparameters:
        self.slope_prior = 1. / (s2 - s1)

        # Define the default number of samples:
        self.nsamples = nsamples

class slope_simulator:
    """
    This example class generates a simulator object that is able to simulate several or 
    single simulations. Simulates same data as the one in the `gen_fake_data()` function.
    """

    def single_simulation(self, parameters):

        # Extract parameters:
        slopes = parameters

        simulated_groups = np.zeros([self.ngroups, self.rows, self.columns])
        # Add the 1/f noise to the group images
        for i in range(self.ngroups):
            _, _, simulated_groups[i, :, :] = utils.generate_detector_ts(self.beta, self.sigma_w, self.sigma_flicker, columns=self.columns, rows = self.rows)

        # Now add the slopes for each pixel we're interested in
        s = 0 # Counter for the slopes we're on
        for pc in self.pixc: 
            for pr in self.pixr:
                # Generate the ramp for this pixel
                ramp = utils.gen_ramp(slope = slopes[s], ngroups=self.ngroups, gain = self.gain, frametime = self.frametime)

                for i in range(self.ngroups):
                    # Now add this ramp to simulations of the 1/f noise to simulate an integration:
                    simulated_groups[i, pr, pc] += ramp[i]

            s+=1

        return simulated_groups[:,self.pixr[0]:self.pixr[-1]+1,self.pixc[0]:self.pixc[-1]+1], simulated_groups

    def several_simulations(self, parameters):

        # Extract parameters:
        all_params = parameters
        nsamples = len(all_params)

        # Define array to store simulations:
        simulations = np.zeros([nsamples, self.length])

        # Lazy loop to do several of these; could apply multi-processing:
        for i in range(nsamples):
            simulations[i,:] = self.single_simulation(all_params[i])

        return simulations

    def __init__(self, length = 1000, ngroups=5, rows=32, columns=512, beta=1 , sigma_w=10 , sigma_flicker=10 , gain=1.42 , frametime=0.902, pixc=[255,256,257], pixr=[16,17,18]):
        self.length = length
        self.ngroups = ngroups
        self.rows = rows
        self.columns = columns
        self.beta = beta
        self.sigma_w = sigma_w
        self.sigma_flicker = sigma_flicker
        self.gain = gain
        self.frametime = frametime
        self.pixc = pixc
        self.pixr = pixr

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