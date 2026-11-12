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
        stardist3d_wa = expand(f"{out_dir}/{{sample_name}}/preprocessing_{{sample_name}}.n5/segmentation/weight_average_stardist3d_plant", zip, sample_name=sample_list),
        stardist3d_wa_plantseg_res = expand(f"{out_dir}/{{sample_name}}/preprocessing_{{sample_name}}.n5/segmentation/weight_average_stardist3d_plant_plantseg_res", zip, sample_name=sample_list),
        affinity_pred = expand(f"{out_dir}/{{sample_name}}/preprocessing_{{sample_name}}.n5/segmentation/affinity_pred", zip, sample_name=sample_list)


rule startdist_3D_weighted_average:
    input:
        f"{out_dir}/{{sample_name}}/multiview_registration_{{sample_name}}.n5/raw/aligned/weighted_average",
    output:
        directory(f"{out_dir}/{{sample_name}}/preprocessing_{{sample_name}}.n5/segmentation/weight_average_stardist3d_plant"),
        directory(f"{out_dir}/{{sample_name}}/preprocessing_{{sample_name}}.n5/segmentation/weight_average_stardist3d_plant_plantseg_res")
    params:
        dapi_chan = lambda wildcards: sample_df.dapi_chan[wildcards.sample_name],
        input_n5_path = f"{out_dir}/{{sample_name}}/multiview_registration_{{sample_name}}.n5",
        output_n5_path = f"{out_dir}/{{sample_name}}/preprocessing_{{sample_name}}.n5",
        log_dir = f"{out_dir}/{{sample_name}}/wa_stardist_plant"
    log: f"{out_dir}/{{sample_name}}/logs/preprocessing/stardist_3d_plant_segmentation.log"
    conda: "stardist_env"
    resources:
        mem_mb="64GB",
        time="02:00:00"
    threads: 32
    shell:
        f"rm -d {{params.output_n5_path}}/segmentation || true;"
        f"rm -d {{params.output_n5_path}} || true;"
        f"mkdir -p {{params.log_dir}};"
        f"if [ \"$HOSTNAME\" = kreshuk-gpu1.embl.de ]; then export CUDA_VISIBLE_DEVICES=4; fi;"
        f"python pipeline_steps/preprocessing/stardist_3d_plant.py {{params.input_n5_path}}  raw/aligned/weighted_average {{params.output_n5_path}} segmentation/weight_average_stardist3d_plant {{params.dapi_chan}} {{params.log_dir}} {{log}};"


rule affinity_pred:
    input:
        f"{out_dir}/{{sample_name}}/multiview_registration_{{sample_name}}.n5/raw/aligned/weighted_average_plantseg_res",
    output:
        directory(f"{out_dir}/{{sample_name}}/preprocessing_{{sample_name}}.n5/segmentation/affinity_pred")
    params:
        input_n5_path = f"{out_dir}/{{sample_name}}/multiview_registration_{{sample_name}}.n5",
        output_n5_path = f"{out_dir}/{{sample_name}}/preprocessing_{{sample_name}}.n5",
        log_dir = f"{out_dir}/{{sample_name}}/affinity_pred"
    log: f"{out_dir}/{{sample_name}}/logs/preprocessing/affinity_pred.log"
    conda: "plant-seg-dev"
    resources:
        time="02:00:00",
        # mem_mb="350GB",
        mem_mb="256GB",
        cpus_per_gpu=16
    threads: 32
    shell:
        f"rm -d {{params.output_n5_path}}/segmentation || true;"
        f"rm -d {{params.output_n5_path}} || true;"
        f"mkdir -p {{params.log_dir}};"
        f"if [ \"$HOSTNAME\" = kreshuk-gpu1.embl.de ]; then export CUDA_VISIBLE_DEVICES=4; fi;"
        f"if [ \"$HOSTNAME\" = kreshuk-gpu3.embl.de ]; then export CUDA_VISIBLE_DEVICES=7; fi;"
        f"python pipeline_steps/preprocessing/plantseg_unet_prediction.py {{params.input_n5_path}} raw/aligned/weighted_average_plantseg_res {{params.output_n5_path}} segmentation/affinity_pred   /g/kreshuk/buglakova/projects/platy_segmentation/plantseg_affinities/plantseg_affogato_affinities_2rois/best_checkpoint.pytorch {{params.log_dir}} {{log}};"