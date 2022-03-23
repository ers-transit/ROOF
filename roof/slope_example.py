import matplotlib.pyplot as plt
import numpy as np

import corner
import abeec 
from abc_classes import example_prior, example_distance, example_simulator

# Read simulated dataset (you can generate a new test dataset 
# by doing import basic_utils; basic_utils.gen_fake_data()):
data = np.loadtxt('slope_data.dat', unpack = True)

# Perform ABC sampling:
samples = abeec.sampler.sample(example_prior(), example_distance(data,distance_fn = 'slopes'), example_simulator(), \
                               M = 3000, N = 300, Delta = 0.1,\
                               verbose = True)

# Extract the 300 posterior samples from the latest particle:
tend = list(samples.keys())[-1]
slopes = samples[tend]['thetas']

# Plot corner plot:
figure = corner.corner(slopes,labels= [f"$\lambda_{{i}}" for i in range(len(slopes))],\
                      quantiles=[0.16,0.5,0.84],\
                      show_titles=True,title_kwargs={"fonstize": 12})


# Plot true values:
true_values = [0]*9 # a, b, sigma
axes = np.array(figure.axes).reshape((9,9))

# Loop over histograms:
for i in range(9):
    ax = axes[i,i]
    ax.axvline(true_values[i], color = 'cornflowerblue')

# Now loop over 2D surfaces:
for yi in range(9):
    for xi in range(yi):
        ax = axes[yi, xi]
        ax.axvline(true_values[xi], color = 'cornflowerblue')
        ax.axhline(true_values[yi], color = 'cornflowerblue')
        ax.plot(true_values[xi], true_values[yi], 'o', mfc = 'cornflowerblue', mec = 'cornflowerblue')

plt.show()
