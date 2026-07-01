"""Deprecated manual EEG viewer script.

Use the FACETpy CLI viewer instead:

    facetpy-run viewer --input output/moosmann/EPICONN-PILOT001_20180402_070226_chunk_001_of_002.edf --viewer-mode mne --show --n-channels 30 --scalings auto

The old direct-MNE viewer is kept below as comments for reference only.
"""

# import mne
#
# import matplotlib
#
# matplotlib.use("QtAgg")
# import matplotlib.pyplot as plt
#
# # Before
# # raw = mne.io.read_raw_egi(
# #     "/home/cfischmei/projects/facetpy_data/rawdata/EEGfMRI20220801_20220801_163809.mff",
# #     preload=True,
# # )
#
# # After
# # raw = mne.io.read_raw_edf(
# #     "/home/cfischmei/projects/facetpy/output/moosmann/EPICONN-PILOT001_20180402_070226_chunk_001_of_002.edf",
# #     preload=True,
# # )
#
# # raw.plot(block=True, scalings="auto", n_channels=30, title="EEG Viewer", show=True)
