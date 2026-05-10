from __future__ import annotations

from pathlib import Path

import pandas as pd

from allensdk.core.brain_observatory_cache import BrainObservatoryCache
import allensdk.brain_observatory.stimulus_info as stim_info


MANIFEST_FILE = Path(
    "/home/maria/Documents/AllenBrainObservatory/brain_observatory_manifest.json"
)


def main() -> None:
    boc = BrainObservatoryCache(manifest_file=str(MANIFEST_FILE))

    print("Loading cell specimen metadata...")
    cells = pd.DataFrame.from_records(boc.get_cell_specimens())
    print(f"Total cells in metadata: {len(cells)}")

    print("Finding Cux2-CreERT2 experiment containers...")
    cux2_ecs = boc.get_experiment_containers(cre_lines=["Cux2-CreERT2"])
    print(f"Cux2 experiment containers: {len(cux2_ecs)}")

    if len(cux2_ecs) == 0:
        raise RuntimeError("No Cux2-CreERT2 experiment containers found.")

    # We try several thresholds, from strict to permissive.
    dsi_thresholds = [0.9, 0.8, 0.7, 0.5, 0.0]

    for ec in cux2_ecs:
        ec_id = int(ec["id"])

        print("\n" + "=" * 80)
        print(f"Trying experiment container: {ec_id}")

        exps = boc.get_ophys_experiments(
            experiment_container_ids=[ec_id],
            stimuli=[stim_info.DRIFTING_GRATINGS],
        )

        if len(exps) == 0:
            print("No drifting gratings experiment for this container. Skipping.")
            continue

        exp = exps[0]
        exp_id = int(exp["id"])

        print(f"Loading drifting gratings ophys experiment: {exp_id}")

        try:
            data_set = boc.get_ophys_experiment_data(exp_id)
        except Exception as e:
            print(f"Could not load experiment {exp_id}: {e}")
            continue

        available_ids = set(map(int, data_set.get_cell_specimen_ids()))
        print(f"Cells physically available in this NWB file: {len(available_ids)}")

        cells_in_dataset = cells[
            cells["cell_specimen_id"].astype(int).isin(available_ids)
        ].copy()

        print(f"Cells matched with metadata: {len(cells_in_dataset)}")

        if cells_in_dataset.empty:
            print("No metadata rows matched this dataset. Skipping.")
            continue

        # p_dg / g_dsi_dg can contain NaNs, so clean them.
        cells_in_dataset["p_dg"] = pd.to_numeric(
            cells_in_dataset["p_dg"], errors="coerce"
        )
        cells_in_dataset["g_dsi_dg"] = pd.to_numeric(
            cells_in_dataset["g_dsi_dg"], errors="coerce"
        )

        print(
            "Cells with significant drifting gratings response "
            f"(p_dg < 0.05): {(cells_in_dataset['p_dg'] < 0.05).sum()}"
        )

        for threshold in dsi_thresholds:
            candidate_cells = cells_in_dataset[
                (cells_in_dataset["p_dg"] < 0.05)
                & (cells_in_dataset["g_dsi_dg"] > threshold)
            ].copy()

            print(
                f"Candidate DSI cells with g_dsi_dg > {threshold}: "
                f"{len(candidate_cells)}"
            )

            if candidate_cells.empty:
                continue

            # Pick strongest direction selective cell.
            candidate_cells = candidate_cells.sort_values(
                "g_dsi_dg", ascending=False
            )

            dsi_cell_id = int(candidate_cells.iloc[0]["cell_specimen_id"])
            p_dg = candidate_cells.iloc[0]["p_dg"]
            g_dsi_dg = candidate_cells.iloc[0]["g_dsi_dg"]

            print("\nFound usable cell!")
            print(f"Experiment container ID: {ec_id}")
            print(f"Ophys experiment ID: {exp_id}")
            print(f"Cell specimen ID: {dsi_cell_id}")
            print(f"p_dg: {p_dg}")
            print(f"g_dsi_dg: {g_dsi_dg}")

            time, raw_traces = data_set.get_fluorescence_traces(
                cell_specimen_ids=[dsi_cell_id]
            )

            print("\nExtracted fluorescence trace:")
            print(f"time shape: {time.shape}")
            print(f"raw_traces shape: {raw_traces.shape}")

            # raw_traces has shape: n_cells x n_timepoints
            trace = raw_traces[0]

            out = pd.DataFrame(
                {
                    "time": time,
                    "fluorescence": trace,
                    "cell_specimen_id": dsi_cell_id,
                    "ophys_experiment_id": exp_id,
                    "experiment_container_id": ec_id,
                }
            )

            out_path = Path("dsi_cell_fluorescence_trace.csv")
            out.to_csv(out_path, index=False)

            print(f"\nSaved trace to: {out_path.resolve()}")
            return

    raise RuntimeError(
        "Could not find any usable DSI cell in Cux2 drifting gratings experiments."
    )


if __name__ == "__main__":
    main()