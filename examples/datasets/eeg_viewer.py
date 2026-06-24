import mne

import matplotlib
matplotlib.use("QtAgg")
import matplotlib.pyplot as plt

#Before
raw = mne.io.read_raw_egi("/home/cfischmei/projects/facetpy_data/rawdata/EEGfMRI20220801_20220801_163809.mff", preload=True)

#After
#raw = mne.io.read_raw_edf("output/corrected_from_set.edf", preload=True)

raw.plot(block=True, scalings="auto", n_channels=30, title="EEG Viewer", show=True)
