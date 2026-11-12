import pandas as pd
import pathlib

cur_dir = pathlib.Path(__file__).parent.resolve()
workdir = cur_dir.parent
workdir: workdir


data_dir = "data/raw"
out_dir = "data/pipeline_output/sample_processing"

sample_df_path = "data_sample_lists/sample_list_confocal.csv"


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

sample_list_gene1 = [sample_name for sample_name in sample_list if (sample_df.loc[sample_name, "gene1_chan"] >= 0)]
sample_list_gene2 = [sample_name for sample_name in sample_list if (sample_df.loc[sample_name, "gene2_chan"] >= 0)]

print("Sample list gene 1")
print(sample_list_gene1)

print("Sample list gene 2")
print(sample_list_gene2)


rule all:
    input:
        mutex_ws = expand(f"{out_dir}/{{sample_name}}/preprocessing_{{sample_name}}.n5/segmentation/mutex_ws", zip, sample_name=sample_list),
        merged_plantseg_res = expand(f"{out_dir}/{{sample_name}}/preprocessing_{{sample_name}}.n5/segmentation/merged_plantseg_res", zip, sample_name=sample_list),
        merged_em_res = expand(f"{out_dir}/{{sample_name}}/preprocessing_{{sample_name}}.n5/segmentation/merged_em_res", zip, sample_name=sample_list),

rule mutex_ws:
    input:
        f"{out_dir}/{{sample_name}}/preprocessing_{{sample_name}}.n5/segmentation/affinity_pred",
    output:
        directory(f"{out_dir}/{{sample_name}}/preprocessing_{{sample_name}}.n5/segmentation/mutex_ws"),
    params:
        log_dir = f"{out_dir}/{{sample_name}}/mutex_ws",
        input_n5_path = f"{out_dir}/{{sample_name}}/preprocessing_{{sample_name}}.n5",
        output_n5_path = f"{out_dir}/{{sample_name}}/preprocessing_{{sample_name}}.n5",
    log: f"{out_dir}/{{sample_name}}/logs/preprocessing/mutex_ws.log"
    conda: "plant-seg-dev"
    resources:
        time="02:00:00",
        cpus_per_task=16,
        mem_mb="350GB"
    threads: 16
    shell:
        f"python pipeline_steps/preprocessing/blockwise_mutex_ws.py {{params.output_n5_path}} segmentation/affinity_pred {{params.output_n5_path}} segmentation/mutex_ws {{params.log_dir}} {{log}};"


rule merge_segmentations:
    input:
        pred = f"{out_dir}/{{sample_name}}/preprocessing_{{sample_name}}.n5/segmentation/affinity_pred",
        mutex_ws = f"{out_dir}/{{sample_name}}/preprocessing_{{sample_name}}.n5/segmentation/mutex_ws",
        stardist = f"{out_dir}/{{sample_name}}/preprocessing_{{sample_name}}.n5/segmentation/weight_average_stardist3d_plant_plantseg_res",
    output:
        directory(f"{out_dir}/{{sample_name}}/preprocessing_{{sample_name}}.n5/segmentation/merged_plantseg_res"),
        directory(f"{out_dir}/{{sample_name}}/preprocessing_{{sample_name}}.n5/segmentation/merged_em_res")
    params:
        output_n5_path = f"{out_dir}/{{sample_name}}/preprocessing_{{sample_name}}.n5",
        log_dir = f"{out_dir}/{{sample_name}}/merge_segmentation"
    log: f"{out_dir}/{{sample_name}}/logs/preprocessing/merge_segmentation.log"
    conda: "plant-seg-dev"
    resources:
        time="02:00:00",
        cpus_per_task=16,
        mem_mb="350GB"
    shell:
        f"python pipeline_steps/preprocessing/merge_segmentation.py {{params.output_n5_path}} segmentation/weight_average_stardist3d_plant_plantseg_res {{params.output_n5_path}} segmentation/affinity_pred {{params.output_n5_path}} segmentation/mutex_ws {{params.output_n5_path}} segmentation/merged {{params.log_dir}} {{log}};"
