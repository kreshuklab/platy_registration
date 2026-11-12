import pandas as pd
import pathlib

cur_dir = pathlib.Path(__file__).parent.resolve()
workdir = cur_dir.parent
workdir: workdir


data_dir = "data/raw"
out_dir = "data/pipeline_output/sample_processing"

sample_df_path = "data/sample_list_confocal.csv"


# Get list of samples from a table with metadata
sample_df = pd.read_csv(sample_df_path).set_index("sample_name", drop=False)
print(sample_df)
sample_list = sample_df.index.values
print("Sample list")
print(sample_list)


# Replace missing gene channels with -1
values = {"gene2_chan": -1, "gene1_chan": -1}
sample_df = sample_df.fillna(value=values)
sample_df = sample_df.astype({"gene2_chan": int, "gene1_chan": int})
print(sample_df)



rule all:
    input:
        cell_regions = expand(f"{out_dir}/{{sample_name}}/preprocessing_{{sample_name}}.n5/segmentation/cell_regions", zip, sample_name=sample_list),
        gene_assignment = expand(f"{out_dir}/{{sample_name}}/gene_assignment/gene_assignment.csv", zip, sample_name=sample_list)

rule grow_cell_regions:
    input:
        f"{out_dir}/{{sample_name}}/multiview_registration_{{sample_name}}.n5/raw/aligned/weighted_average_em_res",
        f"{out_dir}/{{sample_name}}/preprocessing_{{sample_name}}.n5/segmentation/merged_em_res",
        segm_n5_path = f"{out_dir}/{{sample_name}}/preprocessing_{{sample_name}}.n5",
        raw_n5_path = f"{out_dir}/{{sample_name}}/multiview_registration_{{sample_name}}.n5"
    output:
        directory(f"{out_dir}/{{sample_name}}/preprocessing_{{sample_name}}.n5/segmentation/cell_regions"),
        directory(f"{out_dir}/{{sample_name}}/preprocessing_{{sample_name}}.n5/binary_mask"),
        directory(f"{out_dir}/{{sample_name}}/preprocessing_{{sample_name}}.n5/segmentation/merged_em_res_filtered")
    params:
        dapi_chan = lambda wildcards: sample_df.dapi_chan[wildcards.sample_name],
        log_dir = f"{out_dir}/{{sample_name}}/grow_cell_regions"
    log: f"{out_dir}/{{sample_name}}/logs/preprocessing/grow_cell_regions.log"
    resources:
        mem_mb="128GB",
        time="00:30:00"
    conda: "plant-seg-dev"
    shell:
        f"python pipeline_steps/preprocessing/grow_cell_regions.py {{input.raw_n5_path}} raw/aligned/weighted_average_em_res {{input.segm_n5_path}} segmentation/merged_em_res {{input.segm_n5_path}} segmentation/cell_regions binary_mask segmentation/merged_em_res_filtered {{params.dapi_chan}} {{params.log_dir}} {{log}};"


rule assign_gene_signal:
    input:
        f"{out_dir}/{{sample_name}}/multiview_registration_{{sample_name}}.n5/raw/aligned/weighted_average_em_res",
        f"{out_dir}/{{sample_name}}/preprocessing_{{sample_name}}.n5/segmentation/merged_em_res_filtered",
        f"{out_dir}/{{sample_name}}/preprocessing_{{sample_name}}.n5/segmentation/cell_regions",
        segm_n5_path = f"{out_dir}/{{sample_name}}/preprocessing_{{sample_name}}.n5",
        raw_n5_path = f"{out_dir}/{{sample_name}}/multiview_registration_{{sample_name}}.n5",
        spots_dir = f"{out_dir}/{{sample_name}}/spotiflow"
    output:
        f"{out_dir}/{{sample_name}}/gene_assignment/gene_assignment.csv"
    params:
        dapi_chan = lambda wildcards: sample_df.dapi_chan[wildcards.sample_name],
        gene1_chan = lambda wildcards: sample_df.gene1_chan[wildcards.sample_name],
        gene2_chan = lambda wildcards: sample_df.gene2_chan[wildcards.sample_name],
        log_dir = f"{out_dir}/{{sample_name}}/assign_gene_signal"
    log: f"{out_dir}/{{sample_name}}/logs/preprocessing/gene_assignment.log"
    resources:
        mem_mb="128GB",
        time="00:30:00"
    conda: "plant-seg-dev"
    shell:
        f"python pipeline_steps/preprocessing/assign_gene_signal.py {{input.raw_n5_path}} raw/aligned/weighted_average_em_res {{input.segm_n5_path}} segmentation/merged_em_res_filtered segmentation/cell_regions {{output}} {{params.dapi_chan}} {{params.gene1_chan}} {{input.spots_dir}}/gene1/chan_{{params.gene1_chan}}_spots_em_res.csv {{params.gene2_chan}} {{input.spots_dir}}/gene2/chan_{{params.gene2_chan}}_spots_em_res.csv {{params.log_dir}} {{log}};"