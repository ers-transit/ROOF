import os
from astropy.io import fits
import numpy as np
import pdb

class Data:
	'''
	A simple class to hold data
	'''

	def __init__(self, filepath, **kwargs):
		# Assign the filepath to the data
		self.filepath = filepath

		# Load the data from the filepath
		self.load_data(filepath)


	def load_data(self, filepath):
		'''
		Function to load JWST pipeline data prior to ramp fitting
		'''
		# Check if filepath is actually a file
		if not os.path.isfile(filepath):
			raise ValueError('File path not recognised as a file.')

		# Open FITS file
		with fits.open(filepath) as hdul:
			self.phead = hdul[0].header
			self.sci = hdul['SCI'].data
			self.pixdq = hdul['PIXELDQ'].data
			self.grpdq = hdul['GROUPDQ'].data
			self.err = hdul['ERR'].data

		return

	def finish_stage1(self, output_dir):

		# Putting this import here for now...
		from jwst.pipeline.calwebb_detector1 import Detector1Pipeline

		pipeline = Detector1Pipeline()
		# Skip steps that should already be done
		pipeline.group_scale.skip = True
		pipeline.dq_init.skip = True
		pipeline.saturation.skip = True
		pipeline.ipc.skip = True
		pipeline.superbias.skip = True
		# Persistence step is also skipped for TSO observations
		pipeline.persistence.skip = True

		pipeline.save_results = True
		pipeline.output_dir = output_dir
		pipeline.run(self.filepath)

		return


	# Power spectrum

	# ABC method

	# Median method
	def roof_median(self,spec_window):
		removef=np.empty(np.shape(self.sci))
		for integration in np.arange(np.shape(self.sci)[0]):
			for group in np.arange(np.shape(self.sci)[1]):
				test=self.sci[integration,group,:,:]
				sum_row=np.sum(test,axis=1)
				pos_max=np.argmax(sum_row)
				bkg=np.concatenate((test[:np.max((int(pos_max-spec_window),0)),:],test[np.min((int(pos_max+spec_window),np.shape(test)[0])):,:]))
				med=np.median(bkg,axis=0)
				for row in np.arange(np.shape(test)[0]):
					removef[integration,group,row,:]=test[row]-med

		return removef

	# Polynomial fit method