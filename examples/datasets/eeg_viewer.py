import mne

import matplotlib
matplotlib.use("QtAgg")
import matplotlib.pyplot as plt

#Before
#raw = mne.io.read_raw_egi("/home/cfischmei/projects/facetpy_data/rawdata/EEGfMRI20220801_20220801_163809.mff", preload=True)

#After
raw = mne.io.read_raw_edf("/home/cfischmei/projects/facetpy/output/quickstart_chunks/EEGfMRI20220801_20220801_163809_chunk_002_of_002.edf", preload=True)

raw.plot(block=True, scalings="auto", n_channels=30, title="EEG Viewer", show=True)
