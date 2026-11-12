import pandas as pd
import pathlib

cur_dir = pathlib.Path(__file__).parent.resolve()
workdir = cur_dir.parent
workdir: workdir


data_dir = "data/raw"
out_dir = "data/pipeline_output/em_registration"
preprocessing_dir = "data/pipeline_output/sample_processing"

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
# print(sample_df)


intersample_template_n5 = "data/sbem-6dpf-1-whole-segmented-nuclei.n5"
intersample_template_n5_key = "registered/s2_initial_alignment"

em_pcd = "data/em_registration/init_alignment/point_cloud_s2_init_alignment.pcd"
em_features = "data/em_registration/init_alignment/gene_assignment_s2_init_alignment_tissue.csv"
em_neuropil = "data/neuropil_points.csv"

aggregated_expression = "data/em_registration/nonrigid_alignment"



rule all:
    input:
        cpd_nonrigid_em = expand(f"{out_dir}/{{sample_name}}/cpd_nonrigid/nonrigid_reg_em.pcd", zip, sample_name=sample_list)


rule cpd_nonrigid_registration:
    input:
        pcd_path = f"{out_dir}/{{sample_name}}/rigid_registration/point_cloud_rigid.pcd",
        neuropil_path = f"{out_dir}/{{sample_name}}/neuropil_segmentation/neuropil_points.csv"
    output:
        f"{out_dir}/{{sample_name}}/cpd_nonrigid/nonrigid_reg_em.pcd"
    params:
        log_dir = f"{out_dir}/{{sample_name}}/cpd_nonrigid"
    log: f"{out_dir}/{{sample_name}}/logs/cpd_nonrigid_reg_em.log"
    conda: "cvxpy_cpd_env"
    resources:
        time="03:00:00",
        cpus_per_task=16,
        mem_mb="128GB"
    shell:
        f"if [ \"$HOSTNAME\" = kreshuk-gpu1.embl.de ]; then export CUDA_VISIBLE_DEVICES=4; fi;"
        f"echo $CUDA_VISIBLE_DEVICES;"
        f"python pipeline_steps/inter_sample_registration/match_pointclouds.py {em_pcd} {{input.pcd_path}} {{output}} {em_neuropil} {{input.neuropil_path}} {{params.log_dir}} {{log}} --w 0.00001 --beta 100 --lmd 0.1 --maxiter 150;"




rule cpd_nonrigid_registration_fit_points:
    input:
        pcd_path = f"{out_dir}/{{sample_name}}/cpd_nonrigid/nonrigid_reg_em.pcd",
    output:
        f"{out_dir}/{{sample_name}}/cpd_nonrigid_fit_points/nonrigid_reg_em.pcd"
    params:
        log_dir = f"{out_dir}/{{sample_name}}/cpd_nonrigid_fit_points"
    log: f"{out_dir}/{{sample_name}}/logs/cpd_nonrigid_reg_em_fit_points.log"
    conda: "open3d_env"
    resources:
        time="01:00:00",
        cpus_per_task=8,
        mem_mb="32GB"
    shell:
        f"if [ \"$HOSTNAME\" = kreshuk-gpu1.embl.de ]; then export CUDA_VISIBLE_DEVICES=4; fi;"
        f"echo $CUDA_VISIBLE_DEVICES;"
        f"python pipeline_steps/inter_sample_registration/cpd_registration_nonrigid.py {em_pcd} {{input.pcd_path}} {{output}} {{params.log_dir}} {{log}} --w 0.01 --beta 0.01 --lmd 0.01 --maxiter 100;"

