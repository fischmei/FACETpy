import mne
import numpy as np

from facet import Pipeline, AASCorrection, DownSample, EDFExporter
from facet.core import ProcessingContext
from facet.preprocessing import UpSample
raw = mne.io.read_raw_egi(
    "/home/cfischmei/projects/facetpy_data/rawdata/EEGfMRI20220801_20220801_163809.mff",
    preload=True
)

#Load channel names
# print(raw)
# print(raw.ch_names)


#NEED TO ADJUST BUT SEE CHANELS FIRST
status = raw.copy().pick(["TREV"]).get_data()[0]
#print(status)
triggers = np.where((status[1:] > 0) & (status[:-1] == 0))[0] + 1
#print("Number of triggers:", len(triggers))

context = ProcessingContext(raw=raw)
context = context.with_trigger_samples(triggers)
#print("Context has triggers:", context.has_triggers())
#print("Triggers in context:", len(context.get_triggers()))

pipeline = Pipeline([
    UpSample(factor=10),
    AASCorrection(window_size=30),
    DownSample(factor=10),
    EDFExporter(path="output/corrected_from_setMFF.edf", overwrite=True),
])

result = pipeline.run(context)

#print("Done.")
#print("Triggers used:", len(triggers))