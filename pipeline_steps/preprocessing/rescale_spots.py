import pandas as pd
from platy_reg.n5_utils import read_volume, get_attrs
from pathlib import Path

output_folder = Path("data/pipeline_output/sample_processing/St-Mhc--20250113Hyb--ALMF_Stellaris-40X-Sfg_Pl5/spotiflow/gene1")
input_n5 = "data/pipeline_output/sample_processing/St-Mhc--20250113Hyb--ALMF_Stellaris-40X-Sfg_Pl5/multiview_registration_St-Mhc--20250113Hyb--ALMF_Stellaris-40X-Sfg_Pl5.n5"
input_n5_key = "raw/aligned/weighted_average"
channel = 1

out_fname = output_folder/f"chan_{channel}_spots.csv"


spots_df = pd.read_csv(out_fname)
cur_res = get_attrs(input_n5, str(input_n5_key))["resolution"]
print("Resolution at which spots were detected:", cur_res)
em_res = [0.32, 0.32, 0.32]
rescaling_factors = [c / f for f, c in zip(em_res, cur_res)]
print("EM resolution", em_res)
print("Rescaling to EM resolution", rescaling_factors)
for s, col in zip(rescaling_factors, ("z", "y", "x")):
    spots_df[col] = spots_df[col] * s
spots_df.to_csv(output_folder/f"chan_{channel}_spots_em_res.csv", index=False)



spots_df = pd.read_csv(out_fname)
cur_res = get_attrs(input_n5, str(input_n5_key))["resolution"]
print("Resolution at which spots were detected:", cur_res)
plantseg_res = [0.129, 0.129, 0.129]
rescaling_factors = [c / f for f, c in zip(plantseg_res, cur_res)]
print("Plantseg resolution", plantseg_res)
print("Rescaling to plantseg resolution", rescaling_factors)
for s, col in zip(rescaling_factors, ("z", "y", "x")):
    spots_df[col] = spots_df[col] * s
spots_df.to_csv(output_folder/f"chan_{channel}_spots_plantseg_res.csv", index=False)