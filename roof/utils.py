import numpy as np
from stochastic.processes.noise import ColoredNoise

def generate_detector_ts(beta, sigma_w, sigma_flicker, columns = 2048, rows = 512, pixel_time = 10, jump_time = 120, return_image = True, return_time = True, return_white_noise = False):
    """
    Detector generating function | Author: Nestor Espinoza (nespinoza@stsci.edu)
    -----------------------------------------------------------------------------

    This function simulates a JWST detector image and corresponding time-series of the pixel-reads, assuming the noise 
    follows a $1/f^\beta$ power-law in its power spectrum. This assumes the 1/f pattern (and hence the detector reads) 
    go along the columns of the detector.

    Parameters
    ----------
    beta : float
        Power-law index of the PSD of the noise. 
    sigma_w : boolean
        Square-root of the variance of the added Normal-distributed noise process.
    sigma_flicker : float
        Variance of the power-law process in the time-domain. 
   columns : int
        Number of columns of the detector.
    rows : int
        Number of rows of the detector.
    pixel_time : float
        Time it takes to read a pixel along each column in microseconds. Default is 10 microseconds (i.e., like JWST NIR detectors).
    jump_time : float
        Time it takes to jump from one column to the next once all its pixels have been read, in microseconds. Default is 120 microseconds (i.e., like JWST NIR detectors).
    return_image : boolean
        If True, returns an image with the simulated values. Default is False.
    return_time : boolean 
        If True, returns times as well. Default is False.
    return_white_noise : boolean
        If True, returns the white-noise part of the image separately
    Returns
    -------
    times : `numpy.array`
        The time-stamp of the flux values (i.e., at what time since read-out started were they read). Same units as input times (default is microseconds).
    time_series : `numpy.array`
        The actual flux values on each time-stamp (i.e., the pixel counts as they were read in time).
    image : `numpy.array` 
        The image corresponding to the `times` and `time_series`, if `return_image` is set to True.
    """
    # This is the number of "fake pixels" not read during the waiting time between jumps:
    nfake = int(jump_time/pixel_time)

    # First, generate a time series assuming uniform sampling (we will chop it later to accomodate the jump_time):
    CN = ColoredNoise(beta = beta, t = (rows * columns * pixel_time) + columns * jump_time)

    # Get the samples and time-indexes:
    nsamples = rows * columns + (nfake * columns)
    y = CN.sample(nsamples)
    t = CN.times(nsamples)

    # Now remove samples not actually read by the detector due to the wait times. Commented 
    # loop below took 10 secs (!). New pythonic way is the same thing, takes millisecs, and 
    # gets image for free:

    if return_time:
        t_image = t[:-1].reshape((columns, rows + nfake))
        time_image = t_image[:, :rows]
        times = time_image.flatten()

    y_image = y[:-1].reshape((columns, rows + nfake))
    image = y_image[:, :rows]
    time_series = image.flatten()

    # Set process standard-deviation to input sigma:
    time_series = sigma_flicker * (time_series / np.sqrt(np.var(time_series)) )

    # Add normal noise:
    normal_noise = np.random.normal(0., sigma_w, len(time_series))
    time_series = time_series + normal_noise

    if not return_image:
        if not return_time:
            return time_series
        else:
            return times, time_series

    else:

        # Reshape scaled image by sigma_w and sigma_f:
        image = time_series.reshape(image.shape)

        if return_time:

            # Return all:
            if not return_white_noise:

                return times, time_series, image.transpose()

            else:

                wn_image = normal_noise.reshape(image.shape)
                return times, time_series, image.transpose(), wn_image.transpose()

        else:

            if not return_white_noise:

                return time_series, image.transpose()

            else:

                wn_image = normal_noise.reshape(image.shape)
                return time_series, image.transpose(), wn_image.transpose()

def gen_ramp(slope, ngroups, frametime = 0.90200, read_noise = 1., bkg = 0., gain = 1.42, white_noise = False):
    """
    Ramp generator function | Author: Nestor Espinoza (nespinoza@stsci.edu) & Mike Reagan
    -------------------------------------------------------------------------------------
    
    The main idea behind this ramp-generator function is that the number of counts at each up-the-ramp 
    sample is not impacted directly by read-noise --- *reading* each up-the-ramp sample is what generates 
    additional (white-gaussian in this case) noise. In strict terms, this is the same data-generating 
    process as that of describing a cummulative Poisson Process with measurement errors at each time i:
    
    X(i) = T(i) + WN,
    
    with
    
    T(i) = T(i-1) + P(i),
    
    and where
    
    P(i) ~ Poisson(rate)
    WN ~ Normal(0, sigma^2)
    T(0) = 0
    
    While this ramp generator returns values in counts (which is what detectors really measure) and 
    the read-noise is given in counts, all the calculations happen in electrons (input slope is, in 
    fact, also in electrons). So, the rate above will be slope times the frame-time (in seconds).
    
    Inputs
    ------
    
    :param slope: (float)                         
        Slope of the ramp in e-/s.
        
    :param ngroups: (int)
        Number of groups in the ramp.
            
    :param frametime: (float)
        Frame time in seconds

    :param read_noise: (float)
        Read-noise in ADUs of the detector.

    :param bkg: (optional, float)
        Value of the *detector* background, if one wants T(0) distinct from zero. Set to zero by default.
        
    :param gain: (optional, float)
        Gain in e-/ADU. Set to 1.42 (i.e., NIRSpec subarrays)
        
    :param white_noise: (optional, bool)
        Is white-noise going to be included in the ramps? Default is False.
    
    """
    
    
    # Start the arrays that will hold the true number of counts (T) and the actual measured ramp (X):
    T = np.zeros(ngroups)
    X = np.zeros(ngroups)
    X[0] = bkg
    T[0] = bkg
    
    # Now iterate through the process:
    for i in range(1, ngroups):
        
        # Generate Poisson and Gaussian "kicks":
        P = np.random.poisson(slope * frametime) 

        if white_noise:

            WN = np.random.normal(0., read_noise * gain) 

        else:

            WN = 0.
        
        # Kick T and X:
        T[i] = T[i-1] + P
        
        X[i] = T[i] + WN
        
    # X is in electrons --- convert back to counts:
    return X/gain
