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

neuropil_template_n5 = "data/sbem-6dpf-1-whole-segmented-neuropil.n5"
neuropil_template_n5_key = "registered/s2_edited_initial_alignment"

em_pcd = "data/em_registration/init_alignment/point_cloud_s2_init_alignment.pcd"
em_features = "data/em_registration/init_alignment/gene_assignment_s2_init_alignment_tissue.csv"

aggregated_expression = "data/em_registration/nonrigid_alignment"



rule all:
    input:
        elastix_intersample_aligned_rigid = expand(f"{out_dir}/{{sample_name}}/em_registration_{{sample_name}}.n5/template_aligned_rigid/segmentation", zip, sample_name=sample_list),
        apply_elastix_intersample_aligned_rigid = expand(f"{out_dir}/{{sample_name}}/em_registration_{{sample_name}}.n5/template_aligned_rigid/weighted_average", zip, sample_name=sample_list),
        apply_elastix_intersample_aligned_rigid_cell_regions = expand(f"{out_dir}/{{sample_name}}/em_registration_{{sample_name}}.n5/template_aligned_rigid/cell_regions", zip, sample_name=sample_list),
        update_gene_assignment_rigid = expand(f"{out_dir}/{{sample_name}}/rigid_registration/gene_assignment_rigid.csv", zip, sample_name=sample_list),
        pcd_rigid = expand(f"{out_dir}/{{sample_name}}/rigid_registration/point_cloud_rigid.pcd", zip, sample_name=sample_list),
        neuropil_segmentation = expand(f"{out_dir}/{{sample_name}}/em_registration_{{sample_name}}.n5/template_aligned_rigid/neuropil", zip, sample_name=sample_list),


rule elastix_align_intersample_rigid:
    input:
        f"{preprocessing_dir}/{{sample_name}}/preprocessing_{{sample_name}}.n5/segmentation/merged_em_res_filtered",
    output:
        directory(f"{out_dir}/{{sample_name}}/em_registration_{{sample_name}}.n5/template_aligned_rigid/segmentation"),
        f"{out_dir}/{{sample_name}}/elastix_align_intersample_rigid/TransformParameters.0.txt"
        # f"{out_dir}/{{sample_name}}/align_intersample/rigid_registration/TransformParameters.0.txt"
    params:
        preprocessing_n5_path = f"{preprocessing_dir}/{{sample_name}}/preprocessing_{{sample_name}}.n5",
        out_n5_path = f"{out_dir}/{{sample_name}}/em_registration_{{sample_name}}.n5",
        log_dir = f"{out_dir}/{{sample_name}}/elastix_align_intersample_rigid"
    log: f"{out_dir}/{{sample_name}}/logs/elastix_align_intersample_rigid.log"
    conda: "itkelastix"
    resources:
        time="00:30:00",
        cpus_per_task=32,
        mem_mb="256GB"
    shell:
        # f"rm -r {{params.out_n5_path}}/template_aligned_rigid/segmentation;"
        f"rm -d {{params.out_n5_path}}/template_aligned_rigid || :;"
        f"rm -d {{params.out_n5_path}} || :;"
        f"python pipeline_steps/inter_sample_registration/align_rigid_segmentation.py {intersample_template_n5} {intersample_template_n5_key} {{params.preprocessing_n5_path}} segmentation/merged_em_res_filtered {{params.out_n5_path}} template_aligned_rigid/segmentation {{params.log_dir}} {{log}};"


rule apply_elastix_align_intersample_rigid:
    input:
        f"{preprocessing_dir}/{{sample_name}}/multiview_registration_{{sample_name}}.n5/raw/aligned/weighted_average_em_res",
        f"{out_dir}/{{sample_name}}/elastix_align_intersample_rigid/TransformParameters.0.txt"
    output:
        directory(f"{out_dir}/{{sample_name}}/em_registration_{{sample_name}}.n5/template_aligned_rigid/weighted_average"),
    params:
        preprocessing_n5_path = f"{preprocessing_dir}/{{sample_name}}/multiview_registration_{{sample_name}}.n5",
        out_n5_path = f"{out_dir}/{{sample_name}}/em_registration_{{sample_name}}.n5",
        param_map = f"{out_dir}/{{sample_name}}/elastix_align_intersample_rigid/TransformParameters.0.txt",
        log_dir = f"{out_dir}/{{sample_name}}/elastix_apply_intersample_rigid_wa"
    log: f"{out_dir}/{{sample_name}}/logs/align_weighted_average_rigid.log"
    conda: "itkelastix"
    resources:
        time="00:30:00",
        cpus_per_task=32,
        mem_mb="256GB"
    shell:
        f"python pipeline_steps/inter_sample_registration/apply_elastix_transform.py {{params.preprocessing_n5_path}}  raw/aligned/weighted_average_em_res {{params.out_n5_path}} template_aligned_rigid/weighted_average {{params.param_map}} 3 {{params.log_dir}} {{log}};"


rule apply_elastix_align_intersample_rigid_cell_regions:
    input:
        f"{preprocessing_dir}/{{sample_name}}/preprocessing_{{sample_name}}.n5/segmentation/cell_regions",
        f"{out_dir}/{{sample_name}}/elastix_align_intersample_rigid/TransformParameters.0.txt"
    output:
        directory(f"{out_dir}/{{sample_name}}/em_registration_{{sample_name}}.n5/template_aligned_rigid/cell_regions"),
    params:
        preprocessing_n5_path = f"{preprocessing_dir}/{{sample_name}}/preprocessing_{{sample_name}}.n5",
        out_n5_path = f"{out_dir}/{{sample_name}}/em_registration_{{sample_name}}.n5",
        param_map = f"{out_dir}/{{sample_name}}/elastix_align_intersample_rigid/TransformParameters.0.txt",
        log_dir = f"{out_dir}/{{sample_name}}/elastix_apply_intersample_rigid_cr"
    log: f"{out_dir}/{{sample_name}}/logs/align_rigid_cell_regions.log"
    conda: "itkelastix"
    resources:
        time="00:30:00",
        cpus_per_task=32,
        mem_mb="128GB"
    shell:
        f"python pipeline_steps/inter_sample_registration/apply_elastix_transform.py {{params.preprocessing_n5_path}}  segmentation/cell_regions {{params.out_n5_path}} template_aligned_rigid/cell_regions {{params.param_map}} 0 {{params.log_dir}} {{log}};"


rule update_nuclei_coords:
    input:
        f"{out_dir}/{{sample_name}}/em_registration_{{sample_name}}.n5/template_aligned_rigid/segmentation",
        gene_assignment = f"{preprocessing_dir}/{{sample_name}}/gene_assignment/gene_assignment.csv"
    output:
        f"{out_dir}/{{sample_name}}/rigid_registration/gene_assignment_rigid.csv"
    params:
        n5_path = f"{out_dir}/{{sample_name}}/em_registration_{{sample_name}}.n5",
    log: f"{out_dir}/{{sample_name}}/logs/gene_assignment_rigid.log"
    conda: "itkelastix"
    resources:
        time="00:30:00"
    shell:
        f"python pipeline_steps/inter_sample_registration/assign_new_coords.py {{params.n5_path}} template_aligned_rigid/segmentation {{input.gene_assignment}} {{output}} _rigid;"


rule create_point_cloud:
    input:
        f"{out_dir}/{{sample_name}}/em_registration_{{sample_name}}.n5/template_aligned_rigid/segmentation",
        gene_assignment = f"{out_dir}/{{sample_name}}/rigid_registration/gene_assignment_rigid.csv"
    output:
        f"{out_dir}/{{sample_name}}/rigid_registration/point_cloud_rigid.pcd"
    params:
        log_dir = f"{out_dir}/{{sample_name}}/create_pcd",
        n5_path = f"{out_dir}/{{sample_name}}/em_registration_{{sample_name}}.n5"
    log: f"{out_dir}/{{sample_name}}/logs/csv_to_pcd.log"
    conda: "open3d_env"
    resources:
        time="00:15:00",
    shell:
        f"python pipeline_steps/inter_sample_registration/csv_to_pcd.py {{params.n5_path}} template_aligned_rigid/segmentation {{input.gene_assignment}} {{output}} {{params.log_dir}} {{log}};"



rule neuropil_segmentation:
    input:
        f"{out_dir}/{{sample_name}}/em_registration_{{sample_name}}.n5/template_aligned_rigid/cell_regions",
    output:
        directory(f"{out_dir}/{{sample_name}}/em_registration_{{sample_name}}.n5/template_aligned_rigid/neuropil"),
    params:
        input_n5_path = f"{out_dir}/{{sample_name}}/em_registration_{{sample_name}}.n5",
        log_dir = f"{out_dir}/{{sample_name}}/neuropil_segmentation"
    log: f"{out_dir}/{{sample_name}}/logs/neuropil_segmentation.log"
    conda: "itkelastix"
    resources:
        time="00:30:00",
        cpus_per_task=8,
        mem_mb="32GB"
    shell:
        f"python pipeline_steps/inter_sample_registration/segment_neuropil.py  {{params.input_n5_path}} template_aligned_rigid/cell_regions {neuropil_template_n5} {neuropil_template_n5_key} template_aligned_rigid/neuropil {{params.log_dir}} {{log}};"
