import pandas as pd
import pathlib

cur_dir = pathlib.Path(__file__).parent.resolve()

workdir = cur_dir.parent

workdir: workdir

# Directory with raw images
data_path = "CoE_AP151--GATA123_AP002--Hyb20250304--ALMF_Stellaris"


# Set input and output directories
data_dir = f"data/raw/{data_path}"
out_dir = f"data/pipeline_output/check_raw/{data_path}"

# Get list of all lif samples in the input directory
lif_files,  = glob_wildcards(f"{data_dir}/{{lif_file}}.lif")
print("LIF files")
print(lif_files)

# Get list of all vsi samples in the input directory
vsi_files,  = glob_wildcards(f"{data_dir}/{{vsi_file}}.vsi")
print("VSI files")
print(vsi_files)


rule all:
    input: 
        lif_overview = expand(f"{out_dir}/{{lif_file}}/scenes.txt", zip, lif_file=lif_files),
        vsi_overview = expand(f"{out_dir}/{{vsi_file}}/scenes.txt", zip, vsi_file=vsi_files)


rule check_lif:
    input:
        f"{data_dir}/{{lif_file}}.lif"
    output:
        f"{out_dir}/{{lif_file}}/scenes.txt"
    log: f"{out_dir}/{{lif_file}}/lif_info.log"
    conda: "format_convert_env"
    resources:
        time="01:00:00"
    shell:
        f"python pipeline_steps/checkraw/raw_info.py {{input}} {out_dir}/{{wildcards.lif_file}} {{log}}"


rule check_vsi:
    input:
        f"{data_dir}/{{vsi_file}}.vsi"
    output:
        f"{out_dir}/{{vsi_file}}/scenes.txt"
    log: f"{out_dir}/{{vsi_file}}/vsi_info.log"
    conda: "format_convert_env"
    resources:
        time="00:40:00"
    shell:
        f"python pipeline_steps/checkraw/raw_info.py {{input}} {out_dir}/{{wildcards.vsi_file}} {{log}}"