import pandas as pd
import h5py
import numpy as np

# Paths
behavior_path = "/home/maria/Science/data/spontaneous_behaviors/aggression-behavior.parquet"
neural_path = "/home/maria/Science/data/spontaneous_behaviors/aggression-neural.h5"

# Load behavior
behavior = pd.read_parquet(behavior_path).copy()

# Add within-session neural row index
behavior["frame_idx"] = behavior.groupby("session").cumcount()

# Pick one session
session_name = "subject2_recording0"
beh_sess = behavior[behavior["session"] == session_name].copy()

# Load matching neural matrix: shape = (time, neurons)
with h5py.File(neural_path, "r") as f:
    neural = f[session_name][:]

# Auxiliary variable: MoSeq syllable
aux = beh_sess["MoSeq syllable"].to_numpy()

# Optional: coarse auxiliary state too
state = beh_sess["shMoSeq state"].to_numpy()

# Sanity check
assert len(beh_sess) == neural.shape[0] == len(aux)

print("neural shape:", neural.shape)   # (T, N)
print("aux shape:", aux.shape)         # (T,)
print("first aux labels:", aux[:10])

np.save('/home/maria/Science/data/spontaneous_behaviors/made_data/neural_rec.npy', neural)
np.save('/home/maria/Science/data/spontaneous_behaviors/made_data/moseq_syllables_aux.npy', aux)
np.save('/home/maria/Science/data/spontaneous_behaviors/made_data/higher_order_state.npy', state)