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
        spots_gene1 = expand(f"{out_dir}/{{sample_name}}/spotiflow/gene1/", zip, sample_name=sample_list_gene1),
        spots_gene2 = expand(f"{out_dir}/{{sample_name}}/spotiflow/gene2/", zip, sample_name=sample_list_gene2),



rule detect_spots_gene1:
    input:
        f"{out_dir}/{{sample_name}}/multiview_registration_{{sample_name}}.n5/raw/aligned/weighted_average",
        n5_path = f"{out_dir}/{{sample_name}}/multiview_registration_{{sample_name}}.n5"
    output:
        directory(f"{out_dir}/{{sample_name}}/spotiflow/gene1/"),
    params:
        gene1_chan = lambda wildcards: sample_df.gene1_chan[wildcards.sample_name],
        gene1_name = lambda wildcards: sample_df.gene1[wildcards.sample_name],
        log_dir = f"{out_dir}/{{sample_name}}/spotiflow/gene1"
    log: f"{out_dir}/{{sample_name}}/logs/preprocessing/spotiflow_gene1.log"
    resources:
        mem_mb="64GB",
        time="01:00:00"
    conda: "spotiflow"
    shell:
        f"mkdir -p {{params.log_dir}};"
        f"python pipeline_steps/preprocessing/spotiflow_predict_3d.py --input_n5 {{input.n5_path}} --input_n5_key raw/aligned/weighted_average --channel {{params.gene1_chan}}  --output-folder {{output}} --model pipeline_steps/preprocessing/spotiflow_models/smfish3d_finetuned_v2/ >> {{log}} 2>&1;"



rule detect_spots_gene2:
    input:
        f"{out_dir}/{{sample_name}}/multiview_registration_{{sample_name}}.n5/raw/aligned/weighted_average",
        n5_path = f"{out_dir}/{{sample_name}}/multiview_registration_{{sample_name}}.n5"
    output:
        directory(f"{out_dir}/{{sample_name}}/spotiflow/gene2/"),
    params:
        gene2_chan = lambda wildcards: sample_df.gene2_chan[wildcards.sample_name],
        gene2_name = lambda wildcards: sample_df.gene2[wildcards.sample_name],
        log_dir = f"{out_dir}/{{sample_name}}/spotiflow/gene2"
    log: f"{out_dir}/{{sample_name}}/logs/preprocessing/spotiflow_gene2.log"
    resources:
        mem_mb="64GB",
        time="01:00:00"
    conda: "spotiflow"
    shell:
        f"mkdir -p {{params.log_dir}};"
        f"python pipeline_steps/preprocessing/spotiflow_predict_3d.py --input_n5 {{input.n5_path}} --input_n5_key raw/aligned/weighted_average --channel {{params.gene2_chan}}  --output-folder {{output}} --model pipeline_steps/preprocessing/spotiflow_models/smfish3d_finetuned_v2/ >> {{log}} 2>&1;"