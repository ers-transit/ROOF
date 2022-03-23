import matplotlib.pyplot as plt
import numpy as np
from astropy.timeseries import LombScargle
import tqdm

def unwrap_img(img,src_mask = None):
	'''
	Function to convert a 2d image to a 1d time series
	'''
	#time between samples

	#define a source mask to remove spectra pixels if none was provided
	#just mask out a rectangle defined by hand
	if src_mask is None:
		src_mask = np.zeros(np.shape(img),dtype='bool')
		src_mask[8:22,80:450] = True

	#time between samples
	dt = 10 * 1e-6
	
	h,w = np.shape(img)
	times,counts = [],[]
	time = 0
	for i in range(w):
		for j in range(h):
			if not(src_mask[j][i]):
				times.append(time)
				counts.append(img[j][i])
			time += dt
		#time between row readouts
		time += 12 * dt
	return times,counts

#plot power_spectrum
def plot_power_spectrum(images,frequencies = None):
	'''
	Calculate the 1/f power spectrum for a series of images, either supplied or just the raw data
	Takes in a list of images
	'''
	#convert images to a 1d array
	shape = np.shape(images)
	if len(shape) == 4:
		images = np.reshape(images,(shape[0]*shape[1],shape[2],shape[3]))


	if frequencies == None:
		frequencies = np.logspace(1,5,100)
	
	#power spectra for each image to be averaged
	powers = []

	#unwrap each image
	for img in tqdm.tqdm(images):
		times,fluxes = unwrap_img(img)
		#plt.scatter(times,fluxes,s=2)
		power = LombScargle(times,fluxes).power(frequencies)
		powers.append(power)

	power = np.median(powers,axis=0)

	plt.plot(frequencies,power)
	plt.xscale('log')
	plt.yscale('log')
