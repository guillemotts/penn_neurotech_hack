import numpy as np, h5py, json
# load one session
h5 = h5py.File("data/newdata_apr12/smoke_Ani_seed7.h5", "r")
# inspect the raw controller channels (64..75) for ONE trial
cues = json.load(open("data/newdata_apr12/cues_Ani_seed7.json"))
# pick a cue
t_cue_ns = cues["trials"][0]["t_mono_ns"]
# find the sample index in the h5 (depends on your h5 schema — look at keys)
print(list(h5.keys()))
# once you have neural[C,T] and t0_ns for the h5:
# sample_idx = (t_cue_ns - t0_ns) * 32000 // 1e9
# plot ctrl channel (e.g. 64) for [sample_idx - 16000 : sample_idx + 32000]
# look at when the controller actually moves relative to the cue