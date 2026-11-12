import pandas as pd
import pathlib

cur_dir = pathlib.Path(__file__).parent.resolve()
workdir = cur_dir.parent
workdir: workdir
data_dir = "data/raw"
out_dir = "data/pipeline_output/em_registration"
preprocessing_dir = "data/pipeline_output/sample_processing"
platy_browser_logs_dir = "data/pipeline_output/platy_browser_logs"

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
platy_browser_isotropic_n5_key = "s2_isotropic"

em_pcd = "data/em_registration/init_alignment/point_cloud_s2_init_alignment.pcd"
em_features = "data/em_registration/init_alignment/gene_assignment_s2_init_alignment_tissue.csv"
affine_mat = "data/em_registration/init_alignment/affine.txt"

# Platy browser paths
dataset_name = "1.0.1"
mobie_project_folder = "data/platybrowser-smfish-project/data/"
dataset_dir = "data/platybrowser-smfish-project/data/1.0.1"
short_names_df = pd.read_csv(sample_df_path).set_index("short_name", drop=False)
short_names_list = short_names_df.index.values
gene1_names = short_names_df["gene1"]
gene1_chans = short_names_df["gene1_chan"]

df = short_names_df.dropna(subset=['gene2'])
gene2_names = df["gene2"]
gene2_chans = df["gene2_chan"]
short_names_gene2_list = df.index.values

gene_names = [*gene1_names, *gene2_names]
short_names_list = [*short_names_list, *short_names_gene2_list]
gene_chans = [int(chan) for chan in [*gene1_chans, *gene2_chans]]

print(gene_names)
print(short_names_list)
print(gene_chans)



rule all:
    input:
        platy_browser_dapi = expand(f"{platy_browser_logs_dir}/{{short_name}}/platybrowser_add_dapi.txt", zip, short_name=short_names_list),
        platy_browser_gene = expand(f"{platy_browser_logs_dir}/{{short_name}}/platybrowser_add_{{gene_name}}-{{gene_chan}}.txt", zip, short_name=short_names_list, gene_name=gene_names, gene_chan=gene_chans),
        platy_browser_segm = expand(f"{platy_browser_logs_dir}/{{short_name}}/platybrowser_add_segm.txt", zip, short_name=short_names_list),
        transform_pointset = expand(f"{out_dir}/{{sample_name}}/rigid_registration/gene_assignment_rigid.csv", zip, sample_name=sample_list),
        elastix_fixed_pointset = expand(f"{out_dir}/{{sample_name}}/elastix_deformable_pointset_registration/fixed_pointset.txt", zip, sample_name=sample_list),
        elastix_moving_pointset = expand(f"{out_dir}/{{sample_name}}/elastix_deformable_pointset_registration/moving_pointset.txt", zip, sample_name=sample_list),
        elastix_deformable_pointset_apply = expand(f"{out_dir}/{{sample_name}}/em_registration_{{sample_name}}.n5/pointset_aligned_deformable/segmentation", zip, sample_name=sample_list),
        # elastix_deformable_pointset_rough_segm = expand(f"{out_dir}/{{sample_name}}/em_registration_{{sample_name}}.n5/pointset_aligned_deformable_rough/segmentation", zip, sample_name=sample_list),
        elastix_deformable_pointset_fine = expand(f"{out_dir}/{{sample_name}}/em_registration_{{sample_name}}.n5/pointset_aligned_deformable_fine/weighted_average", zip, sample_name=sample_list),
        # elastix_deformable_pointset_rough = expand(f"{out_dir}/{{sample_name}}/em_registration_{{sample_name}}.n5/pointset_aligned_deformable_rough/weighted_average", zip, sample_name=sample_list),
        transform_affine_wa = expand(f"{out_dir}/{{sample_name}}/platy_browser_{{sample_name}}.n5/weighted_average", zip, sample_name=sample_list),
        transform_affine_segm = expand(f"{out_dir}/{{sample_name}}/platy_browser_{{sample_name}}.n5/segmentation", zip, sample_name=sample_list),
        

rule elastix_fixed_pointset:
    input:
        f"{out_dir}/{{sample_name}}/cpd_nonrigid/em_matched_points.pcd"
    output:
        f"{out_dir}/{{sample_name}}/elastix_deformable_pointset_registration/fixed_pointset.txt"
    params:
        log_dir = f"{out_dir}/{{sample_name}}/elastix_deformable_pointset_registration"
    log: f"{out_dir}/{{sample_name}}/logs/elastix_fixed_pointset.log"
    conda: "open3d_env"
    resources:
        time="00:15:00"
    shell:
        f"python pipeline_steps/platybrowser_export/pcd_to_elastix.py {{input}} {{output}} {{params.log_dir}} {{log}};"


rule elastix_moving_pointset:
    input:
        f"{out_dir}/{{sample_name}}/rigid_registration/point_cloud_rigid.pcd"
    output:
        f"{out_dir}/{{sample_name}}/elastix_deformable_pointset_registration/moving_pointset.txt"
    params:
        log_dir = f"{out_dir}/{{sample_name}}/elastix_deformable_pointset_registration"
    log: f"{out_dir}/{{sample_name}}/logs/elastix_moving_pointset.log"
    conda: "open3d_env"
    resources:
        time="00:15:00"
    shell:
        f"python pipeline_steps/platybrowser_export/pcd_to_elastix.py {{input}} {{output}} {{params.log_dir}} {{log}};"


rule elastix_deformable_pointset:
    input:
        f"{out_dir}/{{sample_name}}/em_registration_{{sample_name}}.n5/template_aligned_rigid/segmentation",
        fixed_pointset = f"{out_dir}/{{sample_name}}/elastix_deformable_pointset_registration/fixed_pointset.txt",
        moving_pointset = f"{out_dir}/{{sample_name}}/elastix_deformable_pointset_registration/moving_pointset.txt"
    output:
        temp(directory(f"{out_dir}/{{sample_name}}/em_registration_{{sample_name}}.n5/pointset_aligned_deformable/segmentation")),
        f"{out_dir}/{{sample_name}}/elastix_deformable_pointset_registration/TransformParameters.0.txt",
        f"{out_dir}/{{sample_name}}/elastix_deformable_pointset_registration/TransformParameters.1.txt",
        f"{out_dir}/{{sample_name}}/elastix_deformable_pointset_registration/TransformParameters.2.txt"
    params:
        n5_path = f"{out_dir}/{{sample_name}}/em_registration_{{sample_name}}.n5",
        log_dir = f"{out_dir}/{{sample_name}}/elastix_deformable_pointset_registration"
    log: f"{out_dir}/{{sample_name}}/logs/elastix_deformable_pointset.log"
    conda: "itkelastix"
    resources:
        time="01:00:00",
        cpus_per_task=32,
        mem_mb="256GB"
    shell:
        f"python pipeline_steps/platybrowser_export/elastix_bspline_pointset.py {intersample_template_n5} {intersample_template_n5_key} {{params.n5_path}} template_aligned_rigid/segmentation {{params.n5_path}} pointset_aligned_deformable/segmentation {{input.fixed_pointset}} {{input.moving_pointset}} {{params.log_dir}} {{log}};"


rule apply_elastix_rigid_pointset:
    input:
        f"{out_dir}/{{sample_name}}/em_registration_{{sample_name}}.n5/template_aligned_rigid/weighted_average",
        f"{out_dir}/{{sample_name}}/elastix_deformable_pointset_registration/TransformParameters.0.txt"
    output:
        temp(directory(f"{out_dir}/{{sample_name}}/em_registration_{{sample_name}}.n5/pointset_aligned_rigid/weighted_average")),
    params:
        n5_path = f"{out_dir}/{{sample_name}}/em_registration_{{sample_name}}.n5",
        param_map = f"{out_dir}/{{sample_name}}/elastix_deformable_pointset_registration/TransformParameters.0.txt",
        log_dir = f"{out_dir}/{{sample_name}}/elastix_apply_pointset_rigid"
    log: f"{out_dir}/{{sample_name}}/logs/elastix_apply_pointset_rigid.log"
    conda: "itkelastix"
    resources:
        time="00:30:00",
        cpus_per_task=32,
        mem_mb="64GB"
    shell:
        f"python pipeline_steps/inter_sample_registration/apply_elastix_transform.py {{params.n5_path}}  template_aligned_rigid/weighted_average {{params.n5_path}} pointset_aligned_rigid/weighted_average {{params.param_map}} 3 {{params.log_dir}} {{log}};"



rule apply_elastix_deformable_pointset_rough:
    input:
        f"{out_dir}/{{sample_name}}/em_registration_{{sample_name}}.n5/template_aligned_rigid/weighted_average",
        f"{out_dir}/{{sample_name}}/elastix_deformable_pointset_registration/TransformParameters.1.txt"
    output:
        temp(directory(f"{out_dir}/{{sample_name}}/em_registration_{{sample_name}}.n5/pointset_aligned_deformable_rough/weighted_average")),
    params:
        n5_path = f"{out_dir}/{{sample_name}}/em_registration_{{sample_name}}.n5",
        param_map = f"{out_dir}/{{sample_name}}/elastix_deformable_pointset_registration/TransformParameters.1.txt",
        log_dir = f"{out_dir}/{{sample_name}}/elastix_apply_pointset_deformable_rough"
    log: f"{out_dir}/{{sample_name}}/logs/elastix_apply_pointset_deformable_rough.log"
    conda: "itkelastix"
    resources:
        time="00:30:00",
        cpus_per_task=32,
        mem_mb="64GB"
    shell:
        f"python pipeline_steps/inter_sample_registration/apply_elastix_transform.py {{params.n5_path}}  template_aligned_rigid/weighted_average {{params.n5_path}} pointset_aligned_deformable_rough/weighted_average {{params.param_map}} 3 {{params.log_dir}} {{log}};"


rule apply_elastix_deformable_pointset_fine:
    input:
        f"{out_dir}/{{sample_name}}/em_registration_{{sample_name}}.n5/template_aligned_rigid/weighted_average",
        f"{out_dir}/{{sample_name}}/elastix_deformable_pointset_registration/TransformParameters.2.txt"
    output:
        directory(f"{out_dir}/{{sample_name}}/em_registration_{{sample_name}}.n5/pointset_aligned_deformable_fine/weighted_average"),
    params:
        n5_path = f"{out_dir}/{{sample_name}}/em_registration_{{sample_name}}.n5",
        param_map = f"{out_dir}/{{sample_name}}/elastix_deformable_pointset_registration/TransformParameters.2.txt",
        log_dir = f"{out_dir}/{{sample_name}}/elastix_apply_pointset_deformable_fine"
    log: f"{out_dir}/{{sample_name}}/logs/elastix_apply_pointset_deformable_fine.log"
    conda: "itkelastix"
    resources:
        time="00:30:00",
        cpus_per_task=32,
        mem_mb="64GB"
    shell:
        f"python pipeline_steps/inter_sample_registration/apply_elastix_transform.py {{params.n5_path}} template_aligned_rigid/weighted_average {{params.n5_path}} pointset_aligned_deformable_fine/weighted_average {{params.param_map}} 3 {{params.log_dir}} {{log}};"


rule apply_elastix_rigid_pointset_segm:
    input:
        f"{out_dir}/{{sample_name}}/em_registration_{{sample_name}}.n5/template_aligned_rigid/segmentation",
        f"{out_dir}/{{sample_name}}/elastix_deformable_pointset_registration/TransformParameters.0.txt"
    output:
        temp(directory(f"{out_dir}/{{sample_name}}/em_registration_{{sample_name}}.n5/pointset_aligned_rigid/segmentation")),
    params:
        n5_path = f"{out_dir}/{{sample_name}}/em_registration_{{sample_name}}.n5",
        param_map = f"{out_dir}/{{sample_name}}/elastix_deformable_pointset_registration/TransformParameters.0.txt",
        log_dir = f"{out_dir}/{{sample_name}}/elastix_apply_pointset_rigid_segm"
    log: f"{out_dir}/{{sample_name}}/logs/elastix_apply_pointset_rigid_segm.log"
    conda: "itkelastix"
    resources:
        time="00:30:00",
        cpus_per_task=32,
        mem_mb="64GB"
    shell:
        f"python pipeline_steps/inter_sample_registration/apply_elastix_transform.py {{params.n5_path}} template_aligned_rigid/segmentation {{params.n5_path}} pointset_aligned_rigid/segmentation {{params.param_map}} 0 {{params.log_dir}} {{log}};"


rule apply_elastix_deformable_pointset_rough_segm:
    input:
        f"{out_dir}/{{sample_name}}/em_registration_{{sample_name}}.n5/template_aligned_rigid/segmentation",
        f"{out_dir}/{{sample_name}}/elastix_deformable_pointset_registration/TransformParameters.1.txt"
    output:
        temp(directory(f"{out_dir}/{{sample_name}}/em_registration_{{sample_name}}.n5/pointset_aligned_deformable_rough/segmentation")),
    params:
        n5_path = f"{out_dir}/{{sample_name}}/em_registration_{{sample_name}}.n5",
        param_map = f"{out_dir}/{{sample_name}}/elastix_deformable_pointset_registration/TransformParameters.1.txt",
        log_dir = f"{out_dir}/{{sample_name}}/elastix_apply_pointset_deformable_rough_segm"
    log: f"{out_dir}/{{sample_name}}/logs/elastix_apply_pointset_deformable_rough_segm.log"
    conda: "itkelastix"
    resources:
        time="00:30:00",
        cpus_per_task=32,
        mem_mb="64GB"
    shell:
        f"python pipeline_steps/inter_sample_registration/apply_elastix_transform.py {{params.n5_path}} template_aligned_rigid/segmentation {{params.n5_path}} pointset_aligned_deformable_rough/segmentation {{params.param_map}} 0 {{params.log_dir}} {{log}};"


rule apply_elastix_deformable_pointset_fine_segm:
    input:
        f"{out_dir}/{{sample_name}}/em_registration_{{sample_name}}.n5/template_aligned_rigid/segmentation",
        f"{out_dir}/{{sample_name}}/elastix_deformable_pointset_registration/TransformParameters.2.txt"
    output:
        directory(f"{out_dir}/{{sample_name}}/em_registration_{{sample_name}}.n5/pointset_aligned_deformable_fine/segmentation"),
    params:
        n5_path = f"{out_dir}/{{sample_name}}/em_registration_{{sample_name}}.n5",
        param_map = f"{out_dir}/{{sample_name}}/elastix_deformable_pointset_registration/TransformParameters.2.txt",
        log_dir = f"{out_dir}/{{sample_name}}/elastix_apply_pointset_deformable_fine_segm"
    log: f"{out_dir}/{{sample_name}}/logs/elastix_apply_pointset_deformable_fine_segm.log"
    conda: "itkelastix"
    resources:
        time="00:30:00",
        cpus_per_task=32,
        mem_mb="64GB"
    shell:
        f"python pipeline_steps/inter_sample_registration/apply_elastix_transform.py {{params.n5_path}} template_aligned_rigid/segmentation {{params.n5_path}} pointset_aligned_deformable_fine/segmentation {{params.param_map}} 0 {{params.log_dir}} {{log}};"



rule transform_affine_wa:
    input:
        f"{out_dir}/{{sample_name}}/em_registration_{{sample_name}}.n5/pointset_aligned_deformable_fine/weighted_average",
    output:
        directory(f"{out_dir}/{{sample_name}}/platy_browser_{{sample_name}}.n5/weighted_average"),
    params:
        input_n5_path = f"{out_dir}/{{sample_name}}/em_registration_{{sample_name}}.n5",
        output_n5_path = f"{out_dir}/{{sample_name}}/platy_browser_{{sample_name}}.n5",
        log_dir = f"{out_dir}/{{sample_name}}/transform_affine"
    log: f"{out_dir}/{{sample_name}}/logs/transform_affine_wa.log"
    conda: "itkelastix"
    resources:
        time="01:00:00",
        cpus_per_task=32,
        mem_mb="64GB"
    shell:
        f"rm -d {{params.output_n5_path}} || :;"
        f"python pipeline_steps/platybrowser_export/affine_transform_volumes.py {intersample_template_n5} {platy_browser_isotropic_n5_key} {{params.input_n5_path}}  pointset_aligned_deformable_fine/weighted_average {{params.output_n5_path}} weighted_average {affine_mat} 3 {{params.log_dir}} {{log}};"


rule transform_affine_segm:
    input:
        f"{out_dir}/{{sample_name}}/em_registration_{{sample_name}}.n5/pointset_aligned_deformable_fine/segmentation",
    output:
        directory(f"{out_dir}/{{sample_name}}/platy_browser_{{sample_name}}.n5/segmentation"),
    params:
        input_n5_path = f"{out_dir}/{{sample_name}}/em_registration_{{sample_name}}.n5",
        output_n5_path = f"{out_dir}/{{sample_name}}/platy_browser_{{sample_name}}.n5",
        log_dir = f"{out_dir}/{{sample_name}}/transform_affine_segm"
    log: f"{out_dir}/{{sample_name}}/logs/transform_affine_segm.log"
    conda: "itkelastix"
    resources:
        time="00:30:00",
        cpus_per_task=32,
        mem_mb="64GB"
    shell:
        f"rm -d {{params.output_n5_path}} || :;"
        f"python pipeline_steps/platybrowser_export/affine_transform_volumes.py {intersample_template_n5} {platy_browser_isotropic_n5_key} {{params.input_n5_path}}  pointset_aligned_deformable_fine/segmentation {{params.output_n5_path}} segmentation {affine_mat} 0 {{params.log_dir}} {{log}};"


def construct_n5_name(wildcards):
    short_name = wildcards.short_name
    sample_name = short_names_df.sample_name[short_name]
    return f"{out_dir}/{sample_name}/platy_browser_{sample_name}.n5/weighted_average"


def construct_log_name(wildcards):
    short_name = wildcards.short_name
    sample_name = short_names_df.sample_name[short_name]
    return f"{out_dir}/{sample_name}/logs/platy_browser_add_dapi.log"


rule platybrowser_add_dapi_images:
    input:
        construct_n5_name
    output:
        # directory(f"{dataset_dir}/images/bdv-n5/{{short_name}}_dapi.n5"),
        f"{platy_browser_logs_dir}/{{short_name}}/platybrowser_add_dapi.txt"
    params:
        input_n5_path = lambda wildcards: f"{out_dir}/{short_names_df.sample_name[wildcards.short_name]}/platy_browser_{short_names_df.sample_name[wildcards.short_name]}.n5",
        log_dir = lambda wildcards: f"{out_dir}/{short_names_df.sample_name[wildcards.short_name]}/platybrowser_add_dapi",
        dapi_chan = lambda wildcards: short_names_df.dapi_chan[wildcards.short_name],
    log: f"{platy_browser_logs_dir}/{{short_name}}/platybrowser_add_dapi.log"
    conda: "mobie"
    shell:
        f"python pipeline_steps/platybrowser_export/add_image.py {{params.input_n5_path}} weighted_average/chan_{{params.dapi_chan}} {mobie_project_folder} {dataset_name} HCR_DAPI {{wildcards.short_name}}_dapi {{params.log_dir}} {{log}} >>{{log}} 2>&1;"
        f"rm -r tmp_{dataset_name}_{{wildcards.short_name}}_dapi;"
        f"touch {{output}}"


def construct_log_name(wildcards):
    short_name = wildcards.short_name
    sample_name = short_names_df.sample_name[short_name]
    return f"{out_dir}/{sample_name}/logs/platy_browser_add_dapi.log"


rule platybrowser_add_gene:
    input:
        construct_n5_name
    output:
        f"{platy_browser_logs_dir}/{{short_name}}/platybrowser_add_{{gene_name}}-{{gene_chan}}.txt",
    params:
        input_n5_path = lambda wildcards: f"{out_dir}/{short_names_df.sample_name[wildcards.short_name]}/platy_browser_{short_names_df.sample_name[wildcards.short_name]}.n5",
        log_dir = f"{platy_browser_logs_dir}/{{short_name}}/platybrowser_add_{{gene_name}}--{{short_name}}-{{gene_chan}}",
    log: f"{platy_browser_logs_dir}/{{short_name}}/platybrowser_add_{{gene_name}}--{{short_name}}-{{gene_chan}}.log"
    conda: "mobie"
    shell:
        f"python pipeline_steps/platybrowser_export/add_image.py {{params.input_n5_path}} weighted_average/chan_{{wildcards.gene_chan}} {mobie_project_folder} {dataset_name} HCR {{wildcards.gene_name}}--{{wildcards.short_name}}-{{wildcards.gene_chan}} {{params.log_dir}} {{log}} >>{{log}} 2>&1;"
        f"rm -r tmp_{dataset_name}_{{wildcards.gene_name}}--{{wildcards.short_name}}-{{wildcards.gene_chan}};"
        f"touch {{output}}"


rule platybrowser_add_segm:
    input:
        construct_n5_name
    output:
        # directory(f"{dataset_dir}/images/bdv-n5/{{short_name}}_dapi.n5"),
        f"{platy_browser_logs_dir}/{{short_name}}/platybrowser_add_segm.txt"
    params:
        input_n5_path = lambda wildcards: f"{out_dir}/{short_names_df.sample_name[wildcards.short_name]}/platy_browser_{short_names_df.sample_name[wildcards.short_name]}.n5",
        log_dir = lambda wildcards: f"{out_dir}/{short_names_df.sample_name[wildcards.short_name]}/platybrowser_add_segm",
        dapi_chan = lambda wildcards: short_names_df.dapi_chan[wildcards.short_name],
    log: f"{platy_browser_logs_dir}/{{short_name}}/platybrowser_add_segm.log"
    conda: "mobie"
    shell:
        f"python pipeline_steps/platybrowser_export/add_segmentation.py {{params.input_n5_path}} segmentation {mobie_project_folder} {dataset_name} HCR_segm {{wildcards.short_name}}_segm {{params.log_dir}} {{log}} >>{{log}} 2>&1;"
        f"rm -r tmp_{dataset_name}_{{wildcards.short_name}}_segm;"
        f"touch {{output}}"