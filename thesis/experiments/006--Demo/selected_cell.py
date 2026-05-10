from __future__ import annotations

from pathlib import Path

import pandas as pd

from allensdk.core.brain_observatory_cache import BrainObservatoryCache


MANIFEST_FILE = Path(
    "/home/maria/Documents/AllenBrainObservatory/brain_observatory_manifest.json"
)
#This is the Allen Brain Observatory manifest file path. /home/maria/Documents/AllenBrainObservatory/brain_observatory_manifest.json
#Can you give the dFF plot of the cell 662275084

TARGET_CELL_ID = 662275084


def main() -> None:
    boc = BrainObservatoryCache(manifest_file=str(MANIFEST_FILE))

    print(f"Looking for cell specimen ID: {TARGET_CELL_ID}")

    # Load metadata for all cells
    cells = pd.DataFrame.from_records(boc.get_cell_specimens())

    cell_rows = cells[
        cells["cell_specimen_id"].astype(int) == TARGET_CELL_ID
    ]

    if cell_rows.empty:
        raise RuntimeError(
            f"Cell specimen ID {TARGET_CELL_ID} was not found in Allen cell metadata."
        )

    print("\nCell metadata:")
    print(cell_rows.T)

    # This tells us which experiment container the cell belongs to
    experiment_container_id = int(cell_rows.iloc[0]["experiment_container_id"])

    print(f"\nExperiment container ID: {experiment_container_id}")

    # Get all ophys experiments belonging to this container.
    # A cell may appear in multiple session/stimulus NWB files within the container.
    exps = boc.get_ophys_experiments(
        experiment_container_ids=[experiment_container_id]
    )

    print(f"Found {len(exps)} ophys experiments in this container.")

    if len(exps) == 0:
        raise RuntimeError(
            f"No ophys experiments found for container {experiment_container_id}."
        )

    for exp in exps:
        exp_id = int(exp["id"])
        session_type = exp.get("session_type", "unknown")
        stimuli = exp.get("stimuli", [])

        print("\n" + "=" * 80)
        print(f"Trying ophys experiment ID: {exp_id}")
        print(f"Session type: {session_type}")
        print(f"Stimuli: {stimuli}")

        try:
            data_set = boc.get_ophys_experiment_data(exp_id)
        except Exception as e:
            print(f"Could not load experiment {exp_id}: {e}")
            continue

        available_ids = set(map(int, data_set.get_cell_specimen_ids()))

        if TARGET_CELL_ID not in available_ids:
            print(f"Cell {TARGET_CELL_ID} is not in this NWB file. Skipping.")
            continue

        print(f"\nFound cell {TARGET_CELL_ID} in experiment {exp_id}!")

        time, raw_traces = data_set.get_fluorescence_traces(
            cell_specimen_ids=[TARGET_CELL_ID]
        )

        trace = raw_traces[0]

        print("\nExtracted fluorescence trace:")
        print(f"time shape: {time.shape}")
        print(f"raw_traces shape: {raw_traces.shape}")

        print("\nFirst 20 time points:")
        for t, y in zip(time[:20], trace[:20]):
            print(f"{t:.6f}\t{y:.6f}")

        out = pd.DataFrame(
            {
                "time": time,
                "fluorescence": trace,
                "cell_specimen_id": TARGET_CELL_ID,
                "ophys_experiment_id": exp_id,
                "experiment_container_id": experiment_container_id,
                "session_type": session_type,
            }
        )

        out_path = Path(f"cell_{TARGET_CELL_ID}_fluorescence_trace_exp_{exp_id}.csv")
        out.to_csv(out_path, index=False)

        print(f"\nSaved full trace to:")
        print(out_path.resolve())

        return

    raise RuntimeError(
        f"Cell {TARGET_CELL_ID} was found in metadata, but not in any loaded NWB "
        f"file for experiment container {experiment_container_id}."
    )


if __name__ == "__main__":
    main()