import numpy as np
import matplotlib.pyplot as plt
from roof import utils

beta = 1.
sigma_w = 10. # counts
sigma_flicker = 10. # counts
columns = 512
rows = 32

times, flattened_frame, frame = utils.generate_detector_ts(beta, sigma_w, sigma_flicker, columns = columns, rows = rows)

plt.title('Generated simulated 1/f image')
plt.imshow(frame)
plt.show()

times, flattened_frame, frame, wn_frame = utils.generate_detector_ts(beta, sigma_w, sigma_flicker, columns = columns, rows = rows, return_white_noise = True)

plt.subplot(311)
plt.title('Only white-noise part')
plt.imshow(wn_frame)

plt.subplot(312)
plt.title('Only 1/f part')
plt.imshow(frame - wn_frame)

plt.subplot(313)
plt.title('White-noise + 1/f')
plt.imshow(frame)

plt.show()

plt.title('Flattened time-series of wn + 1/f')
plt.plot(times, flattened_frame)
plt.show()

# Now, generate image with a ramp:
slope = 25. # Slope is in electrons per second
frametime = 0.226 # Frame time is in seconds
ngroups = 5
gain = 1.42 # Gain is in e/adu

# Output is in counts:
ramp = utils.gen_ramp(slope = slope, ngroups = ngroups, gain = gain, frametime = frametime)

group_number = np.arange(ngroups) + 1
plt.plot(group_number, ramp, '.-')
plt.show()

# Now add this ramp to simulationns of the 1/f noise to simulate an integration:
simulated_groups = np.zeros([ngroups, rows, columns])

for i in range(ngroups):

    plt.subplot(ngroups*100 + 10 + (i+1))
    _, _, simulated_groups[i, :, :] = utils.generate_detector_ts(beta, sigma_w, sigma_flicker, columns = columns, rows = rows)

    simulated_groups[i, 15, 300] += ramp[i]
    plt.imshow(simulated_groups[i, :, :])

plt.show()
