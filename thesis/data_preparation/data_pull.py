import os
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from pathlib import Path
from allensdk.core.brain_observatory_cache import BrainObservatoryCache
from collections import defaultdict
from tqdm import tqdm

# ──────────────────────────────────────────────────────────────────────────────
# SETUP
# ──────────────────────────────────────────────────────────────────────────────
load_dotenv()
allen_cache_path = Path(os.environ.get('CAIM_ALLEN_CACHE_PATH'))
assert allen_cache_path.exists(), "CAIM_ALLEN_CACHE_PATH must exist and be set in .env"

stimulus_session_dict = {
    'three_session_A': ['natural_movie_one', 'natural_movie_three'],
    'three_session_B': ['natural_movie_one', 'natural_scenes'],
    'three_session_C': ['natural_movie_one', 'natural_movie_two'],
    'three_session_C2': ['natural_movie_one', 'natural_movie_two']
}

stimulus = 'natural_scenes'
target_num_trials = 50
target_num_frames = 118
total_trials = target_num_trials * target_num_frames  # = 5900

boc = BrainObservatoryCache(
    manifest_file=str(allen_cache_path / 'brain_observatory_manifest.json'))

# ──────────────────────────────────────────────────────────────────────────────
# FUNCTIONS
# ──────────────────────────────────────────────────────────────────────────────
def make_container_dict(boc):
    df = pd.DataFrame(boc.get_ophys_experiments())
    reduced_df = df[['id', 'experiment_container_id', 'session_type']]
    grouped = reduced_df.groupby(['experiment_container_id', 'session_type'])['id'].agg(list).reset_index()
    eid_dict = {}
    for row in grouped.itertuples(index=False):
        cid, stype, ids = row
        if cid not in eid_dict:
            eid_dict[cid] = {}
        eid_dict[cid][stype] = ids[0]
    return eid_dict

def get_valid_session_ids(boc, stimulus='natural_scenes'):
    eid_dict = make_container_dict(boc)
    valid_sessions = []
    for container_id, sessions in eid_dict.items():
        for session_type, eid in sessions.items():
            if session_type in stimulus_session_dict:
                if stimulus in stimulus_session_dict[session_type]:
                    valid_sessions.append(eid)
    return valid_sessions

def get_dff_traces_binary(dff_traces, stim_table, threshold=0.0):
    frame_trials = defaultdict(list)
    for _, row_ in stim_table.iterrows():
        if row_['frame'] != -1:
            start_t, end_t = row_['start'], row_['end']
            frame_idx = row_['frame']
            time_indices = range(start_t, end_t)

            if len(time_indices) == 0:
                trial_vector = np.zeros(dff_traces.shape[0])
            else:
                relevant_traces = dff_traces[:, time_indices]
                trial_vector = np.max(relevant_traces, axis=1)
                trial_vector = (trial_vector > threshold).astype(float)

            frame_trials[frame_idx].append(trial_vector)

    return frame_trials

# ──────────────────────────────────────────────────────────────────────────────
# MAIN LOOP
# ──────────────────────────────────────────────────────────────────────────────
session_ids = get_valid_session_ids(boc, stimulus)[:3]
neural_responses = []

print(f"Processing {len(session_ids)} sessions...")

for sid in tqdm(session_ids):
    try:
        dataset = boc.get_ophys_experiment_data(sid)
        dff_traces = boc.get_ophys_experiment_events(sid)
        stim_table = dataset.get_stimulus_table(stimulus)

        frame_trials = get_dff_traces_binary(dff_traces, stim_table)

        # Initialize response matrix: neurons x (frames × trials)
        num_neurons = dff_traces.shape[0]
        response_matrix = np.full((num_neurons, total_trials), np.nan)

        for frame_idx in range(target_num_frames):
            trials = frame_trials.get(frame_idx, [])
            trials = trials[:target_num_trials]  # in case there's more than 50
            for trial_num, trial in enumerate(trials):
                col_idx = frame_idx * target_num_trials + trial_num
                response_matrix[:, col_idx] = trial

        neural_responses.append((sid, response_matrix))

    except Exception as e:
        print(f"Failed to process session {sid}: {e}")
        continue

# Optionally: stack or save data
all_data = {
    sid: response for sid, response in neural_responses
    if response.shape[1] == total_trials
}
print(f"Collected {len(all_data)} complete sessions.")

# Example save
save_path = allen_cache_path / Path("neural_activity_matrices_")
save_path.mkdir(exist_ok=True)

for sid, matrix in all_data.items():
    np.save(save_path / f"{sid}_neural_responses.npy", matrix)