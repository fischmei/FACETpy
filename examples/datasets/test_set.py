# import mne

# raw = mne.io.read_raw_eeglab("examples/datasets/NiazyFMRI.set", preload=True)

# print(raw)
# print("" \
# "")
# #raw.plot()
# print("" \
# "")
# print(raw.annotations)
# print("" \
# "")
# print(raw.info)
# print("" \
# "")

# print(raw.ch_names)



########check status
# import mne
# import numpy as np

# raw = mne.io.read_raw_eeglab(
# "examples/datasets/NiazyFMRI.set",
# preload=True
# )

# status = raw.copy().pick(["Status"])
# data = status.get_data()[0]

# print("Status min:", data.min())
# print("Status max:", data.max())
# print("Unique values:", np.unique(data)[:50])
# print("Number of unique values:", len(np.unique(data)))


#####check amount


# import mne
# import numpy as np

# raw = mne.io.read_raw_eeglab(
#     "examples/datasets/NiazyFMRI.set",
#     preload=True
# )

# status = raw.copy().pick(["Status"]).get_data()[0]

# rising_edges = np.where((status[1:] == 1) & (status[:-1] == 0))[0] + 1

# print("Number of triggers:", len(rising_edges))
# print("First 20 trigger samples:", rising_edges[:20])
# print("First 20 trigger times in seconds:", rising_edges[:20] / raw.info["sfreq"])


######create trigger manually from status
import mne
print(mne.__version__)

import numpy as np

from facet import Pipeline, AASCorrection, DownSample, EDFExporter
from facet.core import ProcessingContext
from facet.preprocessing import UpSample


raw = mne.io.read_raw_eeglab(
    "examples/datasets/NiazyFMRI.set",
    preload=True
)

status = raw.copy().pick(["Status"]).get_data()[0]
triggers = np.where((status[1:] == 1) & (status[:-1] == 0))[0] + 1

context = ProcessingContext(raw=raw)
context = context.with_trigger_samples(triggers)
print("Context has triggers:", context.has_triggers())
print("Triggers in context:", len(context.get_triggers()))


pipeline = Pipeline([
    UpSample(factor=10),
    AASCorrection(window_size=30),
    DownSample(factor=10),
    EDFExporter(path="output/corrected_from_set2.edf", overwrite=True),
])

result = pipeline.run(context)

print("Done.")
print("Triggers used:", len(triggers))