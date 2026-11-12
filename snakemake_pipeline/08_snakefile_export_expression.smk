import pandas as pd
import pathlib

cur_dir = pathlib.Path(__file__).parent.resolve()
workdir = cur_dir.parent
workdir: workdir


data_dir = "data/raw"
out_dir = "data/pipeline_output/em_registration"
preprocessing_dir = "data/pipeline_output/sample_processing"

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
# print(sample_df)


intersample_template_n5 = "data/sbem-6dpf-1-whole-segmented-nuclei.n5"
intersample_template_n5_key = "registered/s2_initial_alignment"

em_pcd = "data/em_registration/init_alignment/point_cloud_s2_init_alignment.pcd"
em_features = "data/em_registration/init_alignment/gene_assignment_s2_init_alignment_tissue.csv"

aggregated_expression = "data/pipeline_output/aggregated_expression"



rule all:
    input:
        em_ordered_df = expand(f"{out_dir}/{{sample_name}}/aggregated_expression/em_ordered_expression.tsv", zip, sample_name=sample_list),
        sample_ordered_df = expand(f"{out_dir}/{{sample_name}}/aggregated_expression/sample_ordered_expression.tsv", zip, sample_name=sample_list),
        df_collated = f"{aggregated_expression}/expression_matrix_collated.tsv",
        df_aggregated_spots = f"{aggregated_expression}/expression_matrix_aggregated_spots.tsv",
        df_aggregated_mean = f"{aggregated_expression}/expression_matrix_aggregated_mean.tsv"


rule make_expression_table:
    input:
        matched_features = f"{out_dir}/{{sample_name}}/cpd_nonrigid/mapped_features.csv",
        matched_pairs = f"{out_dir}/{{sample_name}}/cpd_nonrigid/matched_pairs.txt"
    output:
        em_ordered_df = f"{out_dir}/{{sample_name}}/aggregated_expression/em_ordered_expression.tsv",
        sample_ordered_df = f"{out_dir}/{{sample_name}}/aggregated_expression/sample_ordered_expression.tsv",
    log: f"{out_dir}/{{sample_name}}/logs/make_expression_table.log"
    params:
        gene1_name = lambda wildcards: sample_df.gene1[wildcards.sample_name],
        gene2_name = lambda wildcards: sample_df.gene2[wildcards.sample_name]
    conda: "cvxpy_cpd_env"
    shell:
        f"python pipeline_steps/platybrowser_export/make_expression_table.py {em_features} {{input.matched_features}} {{input.matched_pairs}} {{output}} {{log}} --gene1_name {{params.gene1_name}} --gene2_name {{params.gene2_name}};"


rule aggregate_expression_result:
    input:
        expand(f"{out_dir}/{{sample_name}}/aggregated_expression/em_ordered_expression.tsv", zip, sample_name=sample_list)
    output:
        df_collated = f"{aggregated_expression}/expression_matrix_collated.tsv",
        df_aggregated_spots = f"{aggregated_expression}/expression_matrix_aggregated_spots.tsv",
        df_aggregated_mean = f"{aggregated_expression}/expression_matrix_aggregated_mean.tsv",
        df_aggregated_sum = f"{aggregated_expression}/expression_matrix_aggregated_sum.tsv"
    params:
        sample_list
    run:
        import pandas as pd
        from pathlib import Path
        import numpy as np

        spots_col_names = []
        sum_col_names = []
        mean_col_names = []

        aggregated_df_spots = pd.DataFrame()
        aggregated_df_mean = pd.DataFrame()
        aggregated_df_sum = pd.DataFrame()
        collated_df = pd.DataFrame()

        for mapped_features, sample_name in zip(input, params[0]):
            short_name = sample_df.short_name[sample_name]
            print("SAMPLE", short_name)

            df = pd.read_csv(mapped_features, sep="\t")
            print(df.columns)
            for col_name in df.columns:
                if col_name == "label_id":
                    collated_df[col_name] = df[col_name]
                    
                else:
                    collated_df[f"{col_name}-{short_name}"] = df[col_name]

                    if col_name.endswith("spots"):
                        spots_col_names.append(col_name)

                    elif col_name.endswith("sum"):
                        sum_col_names.append(col_name)

                    elif col_name.endswith("mean"):
                        mean_col_names.append(col_name)

        
        collated_df.to_csv(output.df_collated, sep="\t", index=False)
        
        spots_col_names = np.unique(spots_col_names)
        sum_col_names = np.unique(sum_col_names)
        mean_col_names = np.unique(mean_col_names)

        aggregated_df_spots["label_id"] = collated_df["label_id"]
        aggregated_df_mean["label_id"] = collated_df["label_id"]

        for col_template in spots_col_names:
            print(col_template)
            cols = [c for c in collated_df.columns if c.startswith(col_template)]
            print(cols)
            gene_name = col_template.replace("_spots", "")
            aggregated_df_spots[gene_name] = np.sum(collated_df[cols], axis=1)

        aggregated_df_spots.to_csv(output.df_aggregated_spots, sep="\t", index=False)

        for col_template in sum_col_names:
            print(col_template)
            cols = [c for c in collated_df.columns if c.startswith(col_template)]
            print(cols)
            gene_name = col_template.replace("_sum", "")
            aggregated_df_sum[gene_name] = np.sum(collated_df[cols], axis=1)

        aggregated_df_sum.to_csv(output.df_aggregated_sum, sep="\t", index=False)

        for col_template in mean_col_names:
            print(col_template)
            cols = [c for c in collated_df.columns if c.startswith(col_template)]
            print(cols)
            gene_name = col_template.replace("_mean", "")
            aggregated_df_mean[gene_name] = np.mean(collated_df[cols], axis=1)
        
        aggregated_df_mean.to_csv(output.df_aggregated_mean, sep="\t", index=False)