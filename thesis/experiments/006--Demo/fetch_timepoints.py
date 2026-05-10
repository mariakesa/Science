
from allensdk.core.brain_observatory_cache import BrainObservatoryCache
from pathlib import Path
import pandas as pd

boc=BrainObservatoryCache(manifest_file=str(Path("/home/maria/Documents/AllenBrainObservatory/brain_observatory_manifest.json")))
data_set=boc.get_ophys_experiment_data(652989705)
stimulus_table=data_set.get_stimulus_table(stimulus_name="natural_scenes")
print(stimulus_table)
