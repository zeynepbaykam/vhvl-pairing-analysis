import pandas as pd
from Bio import PDB
import os
from Bio.PDB.mmcifio import MMCIFIO
from Bio.PDB import MMCIFParser
import gemmi
import subprocess
import json
from tqdm import tqdm
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
from matplotlib.patches import Patch
import umap


def run_arpeggio(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    pdb_files = [f for f in os.listdir(input_dir) if f.endswith(".pdb")]
    cif_files = [f for f in os.listdir(input_dir) if f.endswith(".cif")]
    if len(pdb_files) == len(cif_files):
        print("CIF files already exist for all PDB files. Skipping conversion.")
    else:
        for pdb_file in tqdm(os.listdir(input_dir), desc="Converting PDB to CIF"):
            if pdb_file.endswith(".pdb"):
                structure = gemmi.read_structure(os.path.join(input_dir, pdb_file))
                cif_file = os.path.splitext(pdb_file)[0] + ".cif"
                structure.make_mmcif_document().write_file(os.path.join(input_dir, cif_file))

        for cif_file in tqdm(os.listdir(input_dir), desc="Running Arpeggio"):
            if cif_file.endswith(".cif"):
                json_file = cif_file.replace('.cif', '.json')
                if os.path.exists(os.path.join(output_dir, json_file)):
                    continue  # Skip if output already exists
                input_path = os.path.join(input_dir, cif_file)
                subprocess.run(["pdbe-arpeggio", input_path, "-s", "/H//", "-o", output_dir], capture_output=True)
    return output_dir    


def parse_json(input_dir, output_dir, csv_file_name):
    arpeggio_records = []
    for json_file in tqdm(os.listdir(input_dir), desc="Parsing Arpeggio JSON files"):
        if json_file.endswith(".json"):
            with open(os.path.join(input_dir, json_file), "r") as f:
                data = json.load(f)
                for interaction in data:
                    if interaction["interacting_entities"] == "INTER":
                        record = {
                            "antibody": json_file.split(".")[0],
                            "bgn_chain": interaction["bgn"]["auth_asym_id"],
                            "bgn_resnum": interaction["bgn"]["auth_seq_id"],
                            "bgn_ins_code": interaction["bgn"]["pdbx_PDB_ins_code"],
                            "bgn_resname": interaction["bgn"]["label_comp_id"],
                            "end_chain": interaction["end"]["auth_asym_id"],
                            "end_resnum": interaction["end"]["auth_seq_id"],
                            "end_ins_code": interaction["end"]["pdbx_PDB_ins_code"],
                            "end_resname": interaction["end"]["label_comp_id"],
                            "contact": interaction["contact"],
                            "distance": interaction["distance"],
                            "interacting_entities": interaction["interacting_entities"]
                        }
                        arpeggio_records.append(record)

    arpeggio_df = pd.DataFrame(arpeggio_records)
    arpeggio_explode = arpeggio_df.explode("contact")
    contact_onehot = pd.get_dummies(arpeggio_explode["contact"])
    interactions = pd.concat([arpeggio_explode, contact_onehot], axis=1).groupby(
        ["antibody", "bgn_chain", "bgn_resnum", "bgn_ins_code", "bgn_resname", "end_chain", 
        "end_resnum", "end_ins_code", "end_resname"]).max().drop(["contact", "interacting_entities"], axis=1).reset_index()
    os.makedirs(output_dir, exist_ok=True)
    interactions.to_csv(os.path.join(output_dir, f"{csv_file_name}_arpeggio_contacts.csv"), index=False)
    return interactions



# Sanity checks
# print(bispec_interactions.shape)
# print(bispec_interactions.head())
# print(bispec_interactions["antibody"].nunique())
# print(bispec_interactions["antibody"].str.contains("_.*_.*_").sum())
# bispec_interactions[["bispec", "arm", "cognate"]] = bispec_interactions["antibody"].str.split("_", expand=True)
# bispec_interactions["arm"] = bispec_interactions["arm"].astype(int)
# bispec_interactions["cognate"] = bispec_interactions["cognate"].map({"True": True, "False": False})
# print(bispec_interactions[["bispec", "arm", "cognate"]].head())
# print(bispec_interactions.columns)

def build_feature_matrix(df, threshold):
    df = df.copy()
    df["bgn_resnum"] = df["bgn_resnum"].astype(str) + df["bgn_ins_code"].str.strip().fillna("").astype(str)
    df["end_resnum"] = df["end_resnum"].astype(str) + df["end_ins_code"].str.strip().fillna("").astype(str)
    df["res_pair"] = df.apply(
        lambda row: (
            row["bgn_chain"] + row["bgn_resnum"] + "_" + row["end_chain"] + row["end_resnum"]
            if row["bgn_chain"] == "H"
            else row["end_chain"] + row["end_resnum"] + "_" + row["bgn_chain"] + row["bgn_resnum"]
        ),
        axis=1
    )
    metadata_cols = ["antibody", "bgn_chain", "bgn_resnum", "bgn_ins_code", "bgn_resname", 
                    "end_chain", "end_resnum", "end_ins_code", "end_resname", "distance",
                    "bispecific", "arm", "cognate", "res_pair"]
    interaction_cols = [c for c in df.columns if c not in metadata_cols]
    melted_df = pd.melt(df, id_vars=["antibody", "res_pair"], var_name="interaction_type", value_vars=interaction_cols, value_name="present")
    melted_df = melted_df[melted_df["present"] == True]
    melted_df["feature"] = melted_df["res_pair"] + "_" + melted_df["interaction_type"]
    
    # Deduplicate - same antibody+feature can now appear twice after H-first standardisation
    melted_df = melted_df.groupby(["antibody", "feature"])["present"].max().reset_index()
    
    feature_matrix = melted_df.pivot(columns="feature", index="antibody", values="present").fillna(False).reset_index()
    col_sums = feature_matrix.drop(columns="antibody").sum()
    filtered_features = feature_matrix.loc[:, col_sums[col_sums >= threshold].index]
    filtered_features.index = feature_matrix["antibody"]
    filtered_features = filtered_features.astype(int)
    return filtered_features


def run_pca(df, n):
    scaler = StandardScaler()
    scaled_df = scaler.fit_transform(df)
    pca = PCA(n_components=n)
    principle_components = pca.fit_transform(scaled_df)
    return pca, principle_components


def run_umap(df, n_components=2, n_neighbours=15, min_dist=0.1):
    scaler = StandardScaler()
    scaled_df = scaler.fit_transform(df)
    reducer = umap.UMAP(n_components=n_components, 
                        n_neighbors=n_neighbours, 
                        min_dist=min_dist)
    embedding = reducer.fit_transform(scaled_df)
    return embedding


def plot_figures(explained_variance, principle_components, feature_matrix, embedding):
    # Scree plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    # Individual variance
    ax1.plot(np.arange(1, len(explained_variance) + 1),
            explained_variance, "o-", linewidth=2)
    ax1.set_title("Individual Variance")
    ax1.set_xlabel("Principal Component")
    ax1.set_ylabel("Variance Explained")

    # Cumulative variance
    ax2.plot(np.arange(1, len(explained_variance) + 1),
            np.cumsum(explained_variance), "o-", linewidth=2)
    ax2.axhline(y=0.8, color='r', linestyle='--', label="80%")
    ax2.axhline(y=0.9, color='g', linestyle='--', label="90%")
    ax2.set_title("Cumulative Variance")
    ax2.set_xlabel("Principal Component")
    ax2.set_ylabel("Cumulative Variance Explained")
    ax2.legend()
    plt.tight_layout()
    plt.savefig("scree_plot")
    plt.clf()

    # PC1 vs PC2 coloured by cognate labels
    plt.figure()
    cognate_labels = feature_matrix.index.str.split("_").str[-1]
    colors = cognate_labels.map({"True": "blue", "False": "red"})
    plt.scatter(principle_components[:, 0], principle_components[:, 1], c=colors, alpha=0.5)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("PC1 vs PC2")
    legend_elements = [Patch(facecolor="blue", label="Cognate"),
                       Patch(facecolor="red", label="Non-cognate")]
    plt.legend(handles=legend_elements)
    plt.savefig("PC1_PC2_cognate")
    plt.clf()


    # PC1 vs PC2 coloured by bispecific identity
    plt.figure()
    bispec_identity = feature_matrix.index.str.split("_").str[0]
    unique_bispecs = bispec_identity.unique()
    bispec_to_int = {b:i for i, b in enumerate(unique_bispecs)}
    colors = bispec_identity.map(bispec_to_int)
    plt.scatter(principle_components[:, 0], principle_components[:, 1], c=colors, cmap="hsv", alpha=0.5)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("PC1 vs PC2")
    plt.savefig("PC1_PC2_bispec_identity")
    plt.clf()

    # UMAP coloured by cognate status
    plt.figure()
    cognate_labels = feature_matrix.index.str.split("_").str[-1]
    colors = cognate_labels.map({"True": "blue", "False": "red"})
    plt.scatter(embedding[:, 0], embedding[:, 1], c=colors, alpha=0.5)
    plt.xlabel("UMAP1")
    plt.ylabel("UMAP2")
    plt.title("UMAP coloured by cognate status")
    legend_elements = [Patch(facecolor="blue", label="Cognate"),
                    Patch(facecolor="red", label="Non-cognate")]
    plt.legend(handles=legend_elements)
    plt.savefig("UMAP_cognate.png")
    plt.clf()


def main():
    run_arpeggio("therasabdab_bispec_cognate_noncognate_structures", "bispec_arpeggio_outputs")

    csv_path = "bispec_arpeggio_outputs/therasabdab_bispec_arpeggio_contacts.csv"
    if not os.path.exists(csv_path):
        bispec_interactions = parse_json("bispec_arpeggio_outputs", "bispec_arpeggio_outputs", "therasabdab_bispec")
    else:
        bispec_interactions = pd.read_csv(csv_path)

    feature_matrix = build_feature_matrix(bispec_interactions, 27)
    print(feature_matrix.shape)

    if not os.path.exists("pca_components.npy"):
        pca, principle_components = run_pca(feature_matrix, 200)
        explained_variance = pca.explained_variance_ratio_
        np.save("pca_components.npy", principle_components)
        np.save("pca_variance.npy", pca.explained_variance_ratio_)
    else:
        principle_components = np.load("pca_components.npy")
        explained_variance = np.load("pca_variance.npy")
    
    if not os.path.exists("umap_embedding.npy"):
        embedding = run_umap(feature_matrix)
        np.save("umap_embedding.npy", embedding)
    else:
        embedding = np.load("umap_embedding.npy")
    
    plot_figures(explained_variance, principle_components, feature_matrix, embedding)



    
    # pc_df = pd.DataFrame(principle_components[:, :2], 
    #                  columns=["PC1", "PC2"], 
    #                  index=feature_matrix.index)
    # print(pc_df.nsmallest(5, "PC1"))
    # print(pc_df.nlargest(5, "PC1"))



if __name__ == "__main__":
    main()


#%% Find IMGT positions that are involved on VH-VL interactions across all LICHEN structures
# def interaction_positions(df, chain_id):
#     bgn_chain = df[df["bgn_chain"] == chain_id]
#     end_chain = df[df["end_chain"] == chain_id]
#     return pd.concat([bgn_chain["bgn_resnum"], end_chain["end_resnum"]]).value_counts()

# vh_positions = interaction_positions(interactions, "H").sort_index()
# vl_positions = interaction_positions(interactions, "L").sort_index()

# print(f"VH positions involved in VH-VL interactions across all LICHEN structures:\n{vh_positions}\n")
# print(f"VL positions involved in VH-VL interactions across all LICHEN structures:\n{vl_positions}")

