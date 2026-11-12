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
# print(sample_df)
sample_list = sample_df.index.values
print("Sample list")
print(sample_list)


rule all:
    input:
        aligned_weighted_average = expand(f"{out_dir}/{{sample_name}}/multiview_registration_{{sample_name}}.n5/raw/aligned/weighted_average", zip, sample_name=sample_list),
        aligned_weighted_average_plantseg_res = expand(f"{out_dir}/{{sample_name}}/multiview_registration_{{sample_name}}.n5/raw/aligned/weighted_average_plantseg_res", zip, sample_name=sample_list),
        aligned_weighted_average_em_res = expand(f"{out_dir}/{{sample_name}}/multiview_registration_{{sample_name}}.n5/raw/aligned/weighted_average_em_res", zip, sample_name=sample_list)

        # ventral_ds = expand(f"{out_dir}/{{sample_name}}/multiview_registration_{{sample_name}}.n5/raw/input/ventral", zip, sample_name=sample_list),
        # dorsal_ds = expand(f"{out_dir}/{{sample_name}}/multiview_registration_{{sample_name}}.n5/raw/input/dorsal", zip, sample_name=sample_list),

        # ventral_ds_norm = expand(f"{out_dir}/{{sample_name}}/multiview_registration_{{sample_name}}.n5/raw/input/ventral_norm", zip, sample_name=sample_list),
        # dorsal_ds_norm = expand(f"{out_dir}/{{sample_name}}/multiview_registration_{{sample_name}}.n5/raw/input/dorsal_norm", zip, sample_name=sample_list),

        # ventral_ds_init_aligned = expand(f"{out_dir}/{{sample_name}}/multiview_registration_{{sample_name}}.n5/raw/aligned/ventral/init_alignment", zip, sample_name=sample_list),
        # dorsal_ds_init_aligned = expand(f"{out_dir}/{{sample_name}}/multiview_registration_{{sample_name}}.n5/raw/aligned/dorsal/init_alignment", zip, sample_name=sample_list),
        
        # dorsal_ds_crop = expand(f"{out_dir}/{{sample_name}}/multiview_registration_{{sample_name}}.n5/raw/aligned/dorsal/init_alignment_crop", zip, sample_name=sample_list),
        # ventral_ds_crop = expand(f"{out_dir}/{{sample_name}}/multiview_registration_{{sample_name}}.n5/raw/aligned/ventral/init_alignment_crop", zip, sample_name=sample_list),
        
        # ventral_ds_aligned = expand(f"{out_dir}/{{sample_name}}/multiview_registration_{{sample_name}}.n5/raw/aligned/ventral/rigid_alignment", zip, sample_name=sample_list),
        # dorsal_ds_aligned = expand(f"{out_dir}/{{sample_name}}/multiview_registration_{{sample_name}}.n5/raw/aligned/dorsal/rigid_alignment", zip, sample_name=sample_list),
        
        # ventral_ds_deformable_aligned = expand(f"{out_dir}/{{sample_name}}/multiview_registration_{{sample_name}}.n5/raw/aligned/ventral/deformable_alignment", zip, sample_name=sample_list),
        # dorsal_ds_defromable_aligned = expand(f"{out_dir}/{{sample_name}}/multiview_registration_{{sample_name}}.n5/raw/aligned/dorsal/deformable_alignment", zip, sample_name=sample_list),



rule raw_to_n5:
    input:
    output:
        temp(directory(f"{out_dir}/{{sample_name}}/multiview_registration_{{sample_name}}.n5/raw/input/ventral")),
        temp(directory(f"{out_dir}/{{sample_name}}/multiview_registration_{{sample_name}}.n5/raw/input/dorsal")),
    params:
        ventral_scene = lambda wildcards: sample_df.ventral_scene[wildcards.sample_name],
        dorsal_scene = lambda wildcards: sample_df.dorsal_scene[wildcards.sample_name],
        ventral_path = lambda wildcards: f"{sample_df.data_dir[wildcards.sample_name]}/{sample_df.ventral_filename[wildcards.sample_name]}",
        dorsal_path = lambda wildcards: f"{sample_df.data_dir[wildcards.sample_name]}/{sample_df.dorsal_filename[wildcards.sample_name]}"
    log: f"{out_dir}/{{sample_name}}/logs/multiview_registration/lif_to_n5.log"
    conda: "format_convert_env"
    resources:
        time="02:00:00"
    shell:
        f"rm -r {out_dir}/{{wildcards.sample_name}}/multiview_registration_{{wildcards.sample_name}}.n5;"
        f"python pipeline_steps/multiview_registration/raw_to_n5.py {data_dir}/{{params.ventral_path}} \"{{params.ventral_scene}}\" {out_dir}/{{wildcards.sample_name}}/multiview_registration_{{wildcards.sample_name}}.n5 raw/input/ventral {{log}};"
        f"python pipeline_steps/multiview_registration/raw_to_n5.py {data_dir}/{{params.dorsal_path}} \"{{params.dorsal_scene}}\" {out_dir}/{{wildcards.sample_name}}/multiview_registration_{{wildcards.sample_name}}.n5 raw/input/dorsal {{log}};"


rule normalize_dorsal:
    input:
        f"{out_dir}/{{sample_name}}/multiview_registration_{{sample_name}}.n5/raw/input/dorsal",
    output:
        temp(directory(f"{out_dir}/{{sample_name}}/multiview_registration_{{sample_name}}.n5/raw/input/dorsal_norm"))
    params:
        n5_path = f"{out_dir}/{{sample_name}}/multiview_registration_{{sample_name}}.n5",
        log_dir = f"{out_dir}/{{sample_name}}/normalize_dorsal"
    log: f"{out_dir}/{{sample_name}}/logs/multiview_registration/normalize_input_dorsal.log"
    conda: "format_convert_env"
    resources:
        time="00:30:00",
        mem_mb="128GB"
    shell:
        f"python pipeline_steps/multiview_registration/normalize_input.py {{params.n5_path}}  raw/input/dorsal {{params.n5_path}} raw/input/dorsal_norm {{params.log_dir}} {{log}};"


rule normalize_ventral:
    input:
        f"{out_dir}/{{sample_name}}/multiview_registration_{{sample_name}}.n5/raw/input/ventral",
    output:
        temp(directory(f"{out_dir}/{{sample_name}}/multiview_registration_{{sample_name}}.n5/raw/input/ventral_norm")),
    params:
        n5_path = f"{out_dir}/{{sample_name}}/multiview_registration_{{sample_name}}.n5",
        log_dir = f"{out_dir}/{{sample_name}}/normalize_ventral"
    log: f"{out_dir}/{{sample_name}}/logs/multiview_registration/normalize_input_ventral.log"
    conda: "format_convert_env"
    resources:
        time="00:30:00",
        mem_mb="128GB"
    shell:
        f"python pipeline_steps/multiview_registration/normalize_input.py {{params.n5_path}}  raw/input/ventral {{params.n5_path}} raw/input/ventral_norm {{params.log_dir}} {{log}};"


rule align_ventral_dorsal_init_orient:
    input:
        f"{out_dir}/{{sample_name}}/multiview_registration_{{sample_name}}.n5/raw/input/ventral_norm",
        f"{out_dir}/{{sample_name}}/multiview_registration_{{sample_name}}.n5/raw/input/dorsal_norm",
    output:
        temp(directory(f"{out_dir}/{{sample_name}}/multiview_registration_{{sample_name}}.n5/raw/aligned/ventral/init_alignment")),
        temp(directory(f"{out_dir}/{{sample_name}}/multiview_registration_{{sample_name}}.n5/raw/aligned/dorsal/init_alignment"))
    params:
        dapi_chan = lambda wildcards: sample_df.dapi_chan[wildcards.sample_name],
        n5_path = f"{out_dir}/{{sample_name}}/multiview_registration_{{sample_name}}.n5",
        log_dir = f"{out_dir}/{{sample_name}}/init_orient"
    log: f"{out_dir}/{{sample_name}}/logs/multiview_registration/align_ventral_dorsal_init_orient.log"
    conda: "itkelastix"
    resources:
        time="02:00:00",
        mem_mb="256GB"
    shell:
        f"python pipeline_steps/multiview_registration/align_ventral_dorsal_initial_alignment.py {{params.n5_path}}  raw/input/ventral_norm raw/input/dorsal_norm {{params.n5_path}} raw/aligned/ventral/init_alignment raw/aligned/dorsal/init_alignment {{params.dapi_chan}} {{params.log_dir}} {{log}};"

rule crop_bbox_ventral:
    input:
        f"{out_dir}/{{sample_name}}/multiview_registration_{{sample_name}}.n5/raw/aligned/ventral/init_alignment",
    output:
        temp(directory(f"{out_dir}/{{sample_name}}/multiview_registration_{{sample_name}}.n5/raw/aligned/ventral/init_alignment_crop")),
    params:
        dapi_chan = lambda wildcards: sample_df.dapi_chan[wildcards.sample_name],
        n5_path = f"{out_dir}/{{sample_name}}/multiview_registration_{{sample_name}}.n5",
        log_dir = f"{out_dir}/{{sample_name}}/crop_bbox_ventral"
    log: f"{out_dir}/{{sample_name}}/logs/multiview_registration/crop_bbox_ventral.log"
    conda: "itkelastix"
    resources:
        time="00:20:00",
        mem_mb="128GB"
    shell:
        f"python pipeline_steps/multiview_registration/crop_bbox.py {{params.n5_path}}  raw/aligned/ventral/init_alignment {{params.n5_path}} raw/aligned/ventral/init_alignment_crop {{params.dapi_chan}} {{params.log_dir}} {{log}};"


rule crop_bbox_dorsal:
    input:
        f"{out_dir}/{{sample_name}}/multiview_registration_{{sample_name}}.n5/raw/aligned/dorsal/init_alignment",
    output:
        temp(directory(f"{out_dir}/{{sample_name}}/multiview_registration_{{sample_name}}.n5/raw/aligned/dorsal/init_alignment_crop"))
    params:
        dapi_chan = lambda wildcards: sample_df.dapi_chan[wildcards.sample_name],
        n5_path = f"{out_dir}/{{sample_name}}/multiview_registration_{{sample_name}}.n5",
        log_dir = f"{out_dir}/{{sample_name}}/crop_bbox_dorsal"
    log: f"{out_dir}/{{sample_name}}/logs/multiview_registration/crop_bbox_dorsal.log"
    conda: "itkelastix"
    resources:
        time="00:20:00",
        mem_mb="128GB"
    shell:
        f"python pipeline_steps/multiview_registration/crop_bbox.py {{params.n5_path}}  raw/aligned/dorsal/init_alignment {{params.n5_path}} raw/aligned/dorsal/init_alignment_crop {{params.dapi_chan}} {{params.log_dir}} {{log}};"



rule align_ventral_dorsal_rigid:
    input:
        f"{out_dir}/{{sample_name}}/multiview_registration_{{sample_name}}.n5/raw/aligned/ventral/init_alignment_crop",
        f"{out_dir}/{{sample_name}}/multiview_registration_{{sample_name}}.n5/raw/aligned/dorsal/init_alignment_crop",
    output:
        temp(directory(f"{out_dir}/{{sample_name}}/multiview_registration_{{sample_name}}.n5/raw/aligned/ventral/rigid_alignment")),
        temp(directory(f"{out_dir}/{{sample_name}}/multiview_registration_{{sample_name}}.n5/raw/aligned/dorsal/rigid_alignment"))
    params:
        dapi_chan = lambda wildcards: sample_df.dapi_chan[wildcards.sample_name],
        n5_path = f"{out_dir}/{{sample_name}}/multiview_registration_{{sample_name}}.n5",
        log_dir = f"{out_dir}/{{sample_name}}/ventral_dorsal_rigid"
    log: f"{out_dir}/{{sample_name}}/logs/multiview_registration/align_ventral_dorsal_rigid.log"
    conda: "itkelastix"
    resources:
        time="01:00:00",
        cpus_per_task=32,
        mem_mb="256GB"
    shell:
        f"python pipeline_steps/multiview_registration/align_ventral_dorsal_rigid.py {{params.n5_path}}  raw/aligned/ventral/init_alignment_crop raw/aligned/dorsal/init_alignment_crop {{params.n5_path}} raw/aligned/ventral/rigid_alignment raw/aligned/dorsal/rigid_alignment {{params.dapi_chan}} {{params.log_dir}} {{log}};"


rule align_ventral_dorsal_deformable:
    input:
        f"{out_dir}/{{sample_name}}/multiview_registration_{{sample_name}}.n5/raw/aligned/ventral/rigid_alignment",
        f"{out_dir}/{{sample_name}}/multiview_registration_{{sample_name}}.n5/raw/aligned/dorsal/rigid_alignment",
    output:
        temp(directory(f"{out_dir}/{{sample_name}}/multiview_registration_{{sample_name}}.n5/raw/aligned/ventral/deformable_alignment")),
        temp(directory(f"{out_dir}/{{sample_name}}/multiview_registration_{{sample_name}}.n5/raw/aligned/dorsal/deformable_alignment"))
    params:
        dapi_chan = lambda wildcards: sample_df.dapi_chan[wildcards.sample_name],
        n5_path = f"{out_dir}/{{sample_name}}/multiview_registration_{{sample_name}}.n5",
        log_dir = f"{out_dir}/{{sample_name}}/ventral_dorsal_deformable"
    log: f"{out_dir}/{{sample_name}}/logs/multiview_registration/align_ventral_dorsal_deformable.log"
    conda: "itkelastix"
    resources:
        time="02:00:00",
        cpus_per_task=32,
        mem_mb="256GB"
    shell:
        f"python pipeline_steps/multiview_registration/align_ventral_dorsal_deformable.py deformable_reg pipeline_steps/multiview_registration/ParameterMap_bspline.txt {{params.n5_path}}  raw/aligned/ventral/rigid_alignment raw/aligned/dorsal/rigid_alignment {{params.n5_path}} raw/aligned/ventral/deformable_alignment raw/aligned/dorsal/deformable_alignment {{params.dapi_chan}} {{params.log_dir}} {{log}};"
        

rule align_ventral_dorsal_weighted_average:
    input:
        f"{out_dir}/{{sample_name}}/multiview_registration_{{sample_name}}.n5/raw/aligned/ventral/deformable_alignment",
        f"{out_dir}/{{sample_name}}/multiview_registration_{{sample_name}}.n5/raw/aligned/dorsal/deformable_alignment",
    output:
        directory(f"{out_dir}/{{sample_name}}/multiview_registration_{{sample_name}}.n5/raw/aligned/weighted_average"),
        directory(f"{out_dir}/{{sample_name}}/multiview_registration_{{sample_name}}.n5/raw/aligned/weighted_average_plantseg_res"),
        directory(f"{out_dir}/{{sample_name}}/multiview_registration_{{sample_name}}.n5/raw/aligned/weighted_average_em_res")
    params:
        dapi_chan = lambda wildcards: sample_df.dapi_chan[wildcards.sample_name],
        n5_path = f"{out_dir}/{{sample_name}}/multiview_registration_{{sample_name}}.n5",
        log_dir = f"{out_dir}/{{sample_name}}/ventral_dorsal_weighted_average"
    log: f"{out_dir}/{{sample_name}}/logs/multiview_registration/align_ventral_dorsal_weighted_average.log"
    conda: "itkelastix"
    resources:
        time="00:50:00",
        cpus_per_task=16,
        mem_mb="180GB"
    shell:
        f"python pipeline_steps/multiview_registration/align_ventral_dorsal_weighted_average.py {{params.n5_path}}  raw/aligned/ventral/deformable_alignment raw/aligned/dorsal/deformable_alignment {{params.n5_path}} raw/aligned/weighted_average {{params.dapi_chan}} {{params.log_dir}} {{log}};"
