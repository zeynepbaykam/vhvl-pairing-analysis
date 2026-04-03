import os
import pandas as pd
from lichen import LICHEN
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import scipy.stats as stats
from anarci import anarci
import types
import torch
import torch.nn.functional as F
import math
import numpy as np


def load_and_clean_bispecs(df):
    thera_sabdab_df = pd.read_csv(df)

    cols = [
        "Therapeutic",
        "HeavySequence",
        "HeavySequence(ifbispec)",
        "LightSequence",
        "LightSequence(ifbispec)"
    ]

    df_clean = thera_sabdab_df.dropna(subset=cols).copy()

    mask_not_na_string = (
        (df_clean["HeavySequence"] != "na") &
        (df_clean["HeavySequence(ifbispec)"] != "na") &
        (df_clean["LightSequence"] != "na") &
        (df_clean["LightSequence(ifbispec)"] != "na")
    )

    df_clean = df_clean[mask_not_na_string]

    df_common_vl = df_clean[
        (df_clean["HeavySequence"] != df_clean["HeavySequence(ifbispec)"]) &
        (df_clean["LightSequence"] == df_clean["LightSequence(ifbispec)"])
    ].copy()

    df_no_common_vl = df_clean.drop(df_common_vl.index).copy()

    common_vl_pairs = []

    for _, row in df_common_vl.iterrows():
        name = row["Therapeutic"]
        vl = row["LightSequence"]
        vh1 = row["HeavySequence"]
        vh2 = row["HeavySequence(ifbispec)"]

        common_vl_pairs.append({"bispecific": name, "heavy": vh1, "light": vl})
        common_vl_pairs.append({"bispecific": name, "heavy": vh2, "light": vl})

    df_common_vl = pd.DataFrame(common_vl_pairs)
    return df_no_common_vl, df_common_vl


def create_bispecific_pairs(df, cognate=True):
    vh_vl_pairs = []

    for _, row in df.iterrows():
        name = row["Therapeutic"]
        vl1 = row["LightSequence"]
        vl2 = row["LightSequence(ifbispec)"]
        vh1 = row["HeavySequence"]
        vh2 = row["HeavySequence(ifbispec)"]

        if cognate:
            vh_vl_pairs.append({"bispecific": name, "arm": 1, "cognate": True, "heavy": vh1, "light": vl1})
            vh_vl_pairs.append({"bispecific": name, "arm": 2, "cognate": True, "heavy": vh2, "light": vl2})
        else:
            vh_vl_pairs.append({"bispecific": name, "arm": 1, "cognate": False, "heavy": vh1, "light": vl2})
            vh_vl_pairs.append({"bispecific": name, "arm": 2, "cognate": False, "heavy": vh2, "light": vl1})

    df_vh_vl_pairs = pd.DataFrame(vh_vl_pairs)
    return df_vh_vl_pairs


def load_model():
    lichen_model = LICHEN("LICHEN/Model/Model/model_weights.pt",
                  cpu=True,
                  ncpu=4)
    return lichen_model


def calculate_log_likelihoods(df, model):
    heavy_light = df[["heavy", "light"]]
    log_likelihoods = model.light_log_likelihood(heavy_light)
    assert len(df) == len(log_likelihoods), f"Lengths mismatch: {len(df)} vs {len(log_likelihoods)}"
    df_with_ll = pd.concat([df.reset_index(drop=True), log_likelihoods["log_likelihood"].reset_index(drop=True)], axis=1)
    return df_with_ll


def normalise_likelihood(df):
    all_ll = pd.read_csv(df)

    # Get unique light chains only
    unique_lights = all_ll[["light"]].drop_duplicates().reset_index(drop=True)
    unique_lights["heavy"] = ""

    print(f"Unique light chains: {len(unique_lights)}")
    print(unique_lights.head())

    if not os.path.exists("thera_sabdab_light_baseline_ll.csv"):
        if "lichen_model" not in dir():
            lichen_model = load_model()
        baseline_ll = lichen_model.light_log_likelihood(unique_lights)
        baseline_ll.to_csv("thera_sabdab_light_baseline_ll.csv", index=False)
    else:
        baseline_ll = pd.read_csv("thera_sabdab_light_baseline_ll.csv")

    print(f"Light baseline log likelihood: \n{baseline_ll.head()}\n")

    all_ll = all_ll.merge(
        baseline_ll[["light", "log_likelihood"]].rename(columns={"log_likelihood": "baseline_ll"}),
        on="light", how="left"
    )

    all_ll["normalised_ll"] = all_ll["log_likelihood"] - all_ll["baseline_ll"]
    all_ll.to_csv("thera_sabdab_cognate_noncognate_normalised_ll.csv", index=False)
    print(f"Normalised log likelihood: \n{all_ll.head()}\n")
    return all_ll


def calculate_delta_log_likelihoods(df):
    delta_ll = df.pivot(
        columns="cognate", index=["bispecific", "arm"], values="normalised_ll"
    ).reset_index().rename(
        columns={True: "cognate_ll", False: "noncognate_ll"}
    ).assign(delta_log_likelihood=lambda x: x["cognate_ll"] - x["noncognate_ll"])
    return delta_ll


def calculate_delta_perplexity(df):
    delta_perplexity = df.pivot(
        columns="cognate", index=["bispecific", "arm"], values="perplexity"
    ).reset_index().rename(
        columns={True: "cognate_perplexity", False: "noncognate_perplexity"}
    ).assign(delta_perplexity=lambda x: x["cognate_perplexity"] - x["noncognate_perplexity"])
    return delta_perplexity


def plot_distribution(values, xlabel, title, filename):
    plt.figure()
    plt.hist(values, bins=20)
    plt.xlabel(xlabel)
    plt.ylabel("Frequency")
    plt.title(title)
    plt.savefig(filename)
    plt.close()


def run_analysis(values, metric_name):
    if metric_name == "log_likelihood":
        n_expected = len(values[values > 0])
        direction = "positive"
        alternative = "greater"
    elif metric_name == "perplexity":
        n_expected = len(values[values < 0])
        direction = "negative"
        alternative = "less"
    else:
        raise ValueError(f"Unknown metric_name: {metric_name}. Expected 'log_likelihood' or 'perplexity'")

    n_total = len(values)
    print(f"\n{metric_name}:")
    print(f"{n_expected} out of {n_total} ({n_expected/n_total*100:.1f}%) of bispecific arms have {direction} delta")
    result = stats.binomtest(n_expected, n_total, p=0.5, alternative=alternative)
    print(f"Binomial test p-value: {result.pvalue:.4f}")


def get_imgt_region(position_str):
    try:
        pos = int("".join(filter(lambda x: x.isdigit(), str(position_str))))
    except (ValueError, TypeError):
        return "Unknown"
    if 1 <= pos <= 26:
        return "FWR1"
    elif 27 <= pos <= 38:
        return "CDR1"
    elif 39 <= pos <= 55:
        return "FWR2"
    elif 56 <= pos <= 65:
        return "CDR2"
    elif 66 <= pos <= 104:
        return "FWR3"
    elif 105 <= pos <= 117:
        return "CDR3"
    elif 118 <= pos <= 128:
        return "FWR4"
    else:
        return "Unknown"


def get_imgt_mapping_with_offset(light_seq):
    """Returns mapping and the start/end offset in the raw sequence"""
    result = anarci([("seq1", light_seq)], scheme="imgt", output=False)
    hit = result[1][0][0]
    query_start = hit["query_start"]
    query_end = hit["query_end"]

    numbered = result[0][0][0][0]
    mapping = []
    for (imgt_pos, ins_code), aa in numbered:
        if aa != "-":
            mapping.append({
                "raw_pos": query_start + len(mapping),
                "imgt_pos": str(imgt_pos) + ins_code.strip(),
                "aa": aa,
                "region": get_imgt_region(str(imgt_pos))
            })
    return mapping, query_start, query_end


def _decode_likelihood_per_position(self, src, src_mask, tgt, start_symbol, end_symbol):
    src = src.to(self.device)
    src_mask = src_mask.to(self.device)
    tgt = tgt.to(self.device)
    memory = self.model.encode(src, src_mask)

    ys = torch.ones(1, 1).fill_(start_symbol).type(torch.long).to(self.device)
    per_position_log_probs = []
    i = 0

    while i < tgt.shape[0] + 1:
        memory = memory.to(self.device)
        tgt_mask = (self._generate_square_subsequent_mask(ys.size(0)).type(torch.bool)).to(self.device)

        out = self.model.decode(ys, memory, tgt_mask)
        out = out.transpose(0, 1)
        prob = self.model.generator(out[:, -1])
        token_index = tgt[i + 1]

        if token_index != end_symbol:
            probabilities = F.softmax(prob, dim=-1)
            token_prob = probabilities[:, token_index]
            token_log_prob = torch.log(token_prob.detach() + 1e-20)
            per_position_log_probs.append(token_log_prob.cpu().item())

        next_word = token_index.item()
        ys = torch.cat([ys, torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=0)
        if next_word == end_symbol:
            break
        i += 1

    assert torch.equal(tgt, ys)
    return per_position_log_probs


def likelihood_light_per_position(self, heavy_seq, light_seq):
    self.model.eval()
    try:
        [self.tokenizer.vocab_to_token[resn] for resn in heavy_seq]
    except KeyError as e:
        print(f"Heavy sequence contains an invalid residue: {e}")
        return None
    try:
        [self.tokenizer.vocab_to_token[resn] for resn in light_seq]
    except KeyError as e:
        print(f"Light sequence contains an invalid residue: {e}")
        return None

    src = self.tokenizer.encode(heavy_seq).view(-1, 1)
    num_tokens = src.shape[0]
    src_mask = (torch.zeros(num_tokens, num_tokens)).type(torch.bool)
    tgt = self.tokenizer.encode(light_seq).view(-1, 1)

    return self._decode_likelihood_per_position(
        src, src_mask, tgt,
        start_symbol=self.tokenizer.start_token,
        end_symbol=self.tokenizer.end_token
    )


def light_log_likelihood_per_region(self_lichen, input_df):
    regions = ["FWR1", "CDR1", "FWR2", "CDR2", "FWR3", "CDR3", "FWR4"]
    results = []

    for _, row in input_df.iterrows():
        heavy_seq = row["heavy"]
        light_seq = row["light"]

        per_pos_log_probs = self_lichen.LICHEN.likelihood_light_per_position(heavy_seq, light_seq)

        if per_pos_log_probs is None:
            result = {col: None for col in regions}
        else:
            mapping, query_start, query_end = get_imgt_mapping_with_offset(light_seq)

            if query_start + len(mapping) > len(per_pos_log_probs):
                print(f"Warning: mapping exceeds log_probs length for "
                      f"{row.get('bispecific', 'unknown')}: "
                      f"query_start={query_start}, mapping={len(mapping)}, "
                      f"log_probs={len(per_pos_log_probs)}")
                result = {col: None for col in regions}
            else:
                region_scores = {r: 0.0 for r in regions}
                region_counts = {r: 0 for r in regions}

                for i, pos_info in enumerate(mapping):
                    raw_idx = pos_info["raw_pos"]
                    if raw_idx < len(per_pos_log_probs):
                        region = pos_info["region"]
                        if region in region_scores:
                            region_scores[region] += per_pos_log_probs[raw_idx]
                            region_counts[region] += 1

                result = {}
                for r in regions:
                    if region_counts[r] > 0:
                        result[r] = round(region_scores[r] / region_counts[r], 3)
                    else:
                        result[r] = None

        results.append(result)

    results_df = pd.DataFrame(results)
    return pd.concat([input_df.reset_index(drop=True), results_df], axis=1)


def calculate_per_region_perplexity(per_region_ll):
    """Calculate perplexity from per-region mean log likelihoods"""
    regions = ["FWR1", "CDR1", "FWR2", "CDR2", "FWR3", "CDR3", "FWR4"]
    df = per_region_ll.copy()
    for region in regions:
        if region in df.columns:
            df[f"{region}_perplexity"] = df[region].apply(
                lambda x: round(math.exp(-x), 2) if x is not None and not pd.isna(x) else None
            )
    return df


def calculate_delta_per_region(df, value_cols, label="ll"):
    results = []

    bispec_arms = df[["bispecific", "arm"]].drop_duplicates()

    for _, ba in bispec_arms.iterrows():
        bispec = ba["bispecific"]
        arm = ba["arm"]

        cog_row = df[(df["bispecific"] == bispec) &
                     (df["arm"] == arm) &
                     (df["cognate"] == True)]
        noncog_row = df[(df["bispecific"] == bispec) &
                        (df["arm"] == arm) &
                        (df["cognate"] == False)]

        if len(cog_row) == 0 or len(noncog_row) == 0:
            continue

        record = {"bispecific": bispec, "arm": arm}
        for col in value_cols:
            if col in df.columns:
                cog_val = cog_row[col].values[0]
                noncog_val = noncog_row[col].values[0]
                if pd.isna(cog_val) or pd.isna(noncog_val):
                    record[f"delta_{col}"] = None
                else:
                    record[f"delta_{col}"] = round(float(cog_val) - float(noncog_val), 3)

        results.append(record)

    return pd.DataFrame(results)


def patch_lichen_model(lichen_model):
    """Patch the Heavy2Light instance with per-position methods"""
    lichen_model.LICHEN._decode_likelihood_per_position = types.MethodType(
        _decode_likelihood_per_position, lichen_model.LICHEN)
    lichen_model.LICHEN.likelihood_light_per_position = types.MethodType(
        likelihood_light_per_position, lichen_model.LICHEN)
    print("LICHEN model patched with per-position methods")
    return lichen_model


def calculate_baseline_per_region_ll(lichen_model, per_region_ll):
    """
    Calculate baseline per-region log likelihood by running LICHEN with empty heavy chain.
    Returns per-region scores for each unique light chain.
    """
    regions = ["FWR1", "CDR1", "FWR2", "CDR2", "FWR3", "CDR3", "FWR4"]

    unique_lights = per_region_ll[["light"]].drop_duplicates().reset_index(drop=True)
    unique_lights["heavy"] = ""
    unique_lights["bispecific"] = "baseline"
    unique_lights["arm"] = 1
    unique_lights["cognate"] = True

    print(f"Computing baseline per-region LL for {len(unique_lights)} unique light chains...")
    baseline = light_log_likelihood_per_region(lichen_model, unique_lights)

    return baseline[["light"] + regions].rename(
        columns={r: f"{r}_baseline" for r in regions}
    )


def normalise_per_region_ll(per_region_ll, baseline_per_region):
    """Subtract baseline per-region LL from paired per-region LL."""
    regions = ["FWR1", "CDR1", "FWR2", "CDR2", "FWR3", "CDR3", "FWR4"]

    df = per_region_ll.merge(baseline_per_region, on="light", how="left")

    for r in regions:
        df[f"{r}_normalised"] = df[r] - df[f"{r}_baseline"]

    return df


def calculate_normalised_per_region_perplexity(normalised_per_region_ll):
    """Calculate perplexity from normalised per-region log likelihoods."""
    regions = ["FWR1", "CDR1", "FWR2", "CDR2", "FWR3", "CDR3", "FWR4"]
    df = normalised_per_region_ll.copy()

    for r in regions:
        col = f"{r}_normalised"
        if col in df.columns:
            df[f"{r}_normalised_perplexity"] = df[col].apply(
                lambda x: round(math.exp(-x), 2) if x is not None and not pd.isna(x) else None
            )
    return df


## Statistical tests ##
def ttest_per_region_cognate_noncognate(per_region_df, metric="ll"):
    """
    Compare per-region scores between cognate and non-cognate structures.
    Runs both one-tailed and two-tailed paired t-tests.
    metric: 'll' for log likelihood, 'perplexity' for perplexity
    """
    from statsmodels.stats.multitest import multipletests

    if metric == "ll":
        regions = ["FWR1", "CDR1", "FWR2", "CDR2", "FWR3", "CDR3", "FWR4"]
        one_tailed_alternative = "greater"
    else:
        regions = [f"{r}_perplexity" for r in ["FWR1", "CDR1", "FWR2", "CDR2", "FWR3", "CDR3", "FWR4"]]
        one_tailed_alternative = "less"

    cognate_df = per_region_df[per_region_df["cognate"] == True]
    noncognate_df = per_region_df[per_region_df["cognate"] == False]

    cognate_sorted = cognate_df.sort_values(["bispecific", "arm"]).reset_index(drop=True)
    noncognate_sorted = noncognate_df.sort_values(["bispecific", "arm"]).reset_index(drop=True)

    results = []
    p_values_onetailed = []
    p_values_twotailed = []

    for region in regions:
        cog_vals = cognate_sorted[region].dropna()
        noncog_vals = noncognate_sorted[region].dropna()

        valid_idx = cog_vals.index.intersection(noncog_vals.index)
        cog_vals = cog_vals.loc[valid_idx]
        noncog_vals = noncog_vals.loc[valid_idx]

        t_stat_two, p_two = stats.ttest_rel(cog_vals, noncog_vals, alternative="two-sided")
        t_stat_one, p_one = stats.ttest_rel(cog_vals, noncog_vals, alternative=one_tailed_alternative)

        p_values_onetailed.append(p_one)
        p_values_twotailed.append(p_two)

        results.append({
            "region": region,
            "n_pairs": len(cog_vals),
            "cognate_mean": round(cog_vals.mean(), 3),
            "cognate_std": round(cog_vals.std(), 3),
            "cognate_median": round(cog_vals.median(), 3),
            "noncognate_mean": round(noncog_vals.mean(), 3),
            "noncognate_std": round(noncog_vals.std(), 3),
            "noncognate_median": round(noncog_vals.median(), 3),
            "mean_difference": round(cog_vals.mean() - noncog_vals.mean(), 3),
            "t_statistic_twotailed": round(t_stat_two, 3),
            "t_statistic_onetailed": round(t_stat_one, 3),
            "p_value_twotailed": round(p_two, 4),
            "p_value_onetailed": round(p_one, 4),
        })

    results_df = pd.DataFrame(results)

    _, p_adj_two, _, _ = multipletests(p_values_twotailed, method="fdr_bh")
    _, p_adj_one, _, _ = multipletests(p_values_onetailed, method="fdr_bh")

    results_df["p_adjusted_twotailed"] = p_adj_two.round(4)
    results_df["p_adjusted_onetailed"] = p_adj_one.round(4)
    results_df["significant_twotailed"] = results_df["p_adjusted_twotailed"] < 0.05
    results_df["significant_onetailed"] = results_df["p_adjusted_onetailed"] < 0.05

    return results_df


def ttest_normalised_per_region(normalised_per_region_df, metric="ll"):
    """
    Paired t-test on normalised per-region scores between cognate and non-cognate.
    """
    from statsmodels.stats.multitest import multipletests
    regions = ["FWR1", "CDR1", "FWR2", "CDR2", "FWR3", "CDR3", "FWR4"]

    if metric == "ll":
        cols = [f"{r}_normalised" for r in regions]
        one_tailed_alternative = "greater"
    else:
        cols = [f"{r}_normalised_perplexity" for r in regions]
        one_tailed_alternative = "less"

    cognate_df = normalised_per_region_df[normalised_per_region_df["cognate"] == True]\
        .sort_values(["bispecific", "arm"]).reset_index(drop=True)
    noncognate_df = normalised_per_region_df[normalised_per_region_df["cognate"] == False]\
        .sort_values(["bispecific", "arm"]).reset_index(drop=True)

    results = []
    p_values_two = []
    p_values_one = []

    for col in cols:
        cog_vals = cognate_df[col].dropna()
        noncog_vals = noncognate_df[col].dropna()

        valid_idx = cog_vals.index.intersection(noncog_vals.index)
        cog_vals = cog_vals.loc[valid_idx]
        noncog_vals = noncog_vals.loc[valid_idx]

        t_two, p_two = stats.ttest_rel(cog_vals, noncog_vals, alternative="two-sided")
        t_one, p_one = stats.ttest_rel(cog_vals, noncog_vals, alternative=one_tailed_alternative)

        p_values_two.append(p_two)
        p_values_one.append(p_one)

        results.append({
            "region": col,
            "n_pairs": len(cog_vals),
            "cognate_mean": round(cog_vals.mean(), 3),
            "cognate_std": round(cog_vals.std(), 3),
            "cognate_median": round(cog_vals.median(), 3),
            "noncognate_mean": round(noncog_vals.mean(), 3),
            "noncognate_std": round(noncog_vals.std(), 3),
            "noncognate_median": round(noncog_vals.median(), 3),
            "mean_difference": round(cog_vals.mean() - noncog_vals.mean(), 3),
            "t_statistic_twotailed": round(t_two, 3),
            "t_statistic_onetailed": round(t_one, 3),
            "p_value_twotailed": round(p_two, 4),
            "p_value_onetailed": round(p_one, 4),
        })

    results_df = pd.DataFrame(results)
    _, p_adj_two, _, _ = multipletests(p_values_two, method="fdr_bh")
    _, p_adj_one, _, _ = multipletests(p_values_one, method="fdr_bh")
    results_df["p_adjusted_twotailed"] = p_adj_two.round(4)
    results_df["p_adjusted_onetailed"] = p_adj_one.round(4)
    results_df["significant_twotailed"] = results_df["p_adjusted_twotailed"] < 0.05
    results_df["significant_onetailed"] = results_df["p_adjusted_onetailed"] < 0.05

    return results_df


## Plotting ##
def plot_per_region_boxplots(per_region_df, region_stats_df, metric="ll"):
    """
    Box-whisker plots for per-region cognate vs non-cognate comparison.
    """
    if metric == "ll":
        regions = ["FWR1", "CDR1", "FWR2", "CDR2", "FWR3", "CDR3", "FWR4"]
        ylabel = "Mean Log Likelihood per Residue"
        title = "Per-region Log Likelihood: Cognate vs Non-cognate"
        filename = "per_region_ll_boxplots.png"
    else:
        regions = [f"{r}_perplexity" for r in ["FWR1", "CDR1", "FWR2", "CDR2", "FWR3", "CDR3", "FWR4"]]
        ylabel = "Perplexity"
        title = "Per-region Perplexity: Cognate vs Non-cognate"
        filename = "per_region_perplexity_boxplots.png"

    cognate_df = per_region_df[per_region_df["cognate"] == True]
    noncognate_df = per_region_df[per_region_df["cognate"] == False]

    fig, axes = plt.subplots(1, 7, figsize=(22, 6))
    fig.suptitle(title, fontsize=14, fontweight="bold", y=1.02)

    for idx, (ax, region) in enumerate(zip(axes, regions)):
        cog_vals = cognate_df[region].dropna()
        noncog_vals = noncognate_df[region].dropna()

        bp = ax.boxplot([cog_vals, noncog_vals],
                       patch_artist=True,
                       widths=0.4,
                       medianprops=dict(color="black", linewidth=2),
                       whiskerprops=dict(linewidth=1.5),
                       capprops=dict(linewidth=1.5),
                       flierprops=dict(marker="o", markersize=2, alpha=0.3))

        bp["boxes"][0].set_facecolor("#AED6F1")
        bp["boxes"][0].set_edgecolor("#1A5276")
        bp["boxes"][1].set_facecolor("#FADBD8")
        bp["boxes"][1].set_edgecolor("#922B21")

        np.random.seed(42)
        jitter = 0.08
        ax.scatter(np.random.normal(1, jitter, size=len(cog_vals)),
                  cog_vals, alpha=0.3, s=8, color="#1A5276", zorder=3)
        ax.scatter(np.random.normal(2, jitter, size=len(noncog_vals)),
                  noncog_vals, alpha=0.3, s=8, color="#922B21", zorder=3)

        region_label = region.replace("_perplexity", "") if metric == "perplexity" else region
        ax.set_title(region_label, fontsize=11, fontweight="bold")
        ax.set_xticks([1, 2])
        ax.set_xticklabels(["Cognate", "Non-\ncognate"], fontsize=8)
        ax.set_ylabel(ylabel if idx == 0 else "", fontsize=9)
        ax.tick_params(axis="y", labelsize=8)

        p_adj = region_stats_df.loc[region_stats_df["region"] == region,
                                    "p_adjusted_onetailed"].values[0]

        if p_adj < 0.001:
            p_text = "p<0.001***"
        elif p_adj < 0.01:
            p_text = f"p={p_adj:.3f}**"
        elif p_adj < 0.05:
            p_text = f"p={p_adj:.3f}*"
        else:
            p_text = f"p={p_adj:.3f}\nns"

        current_ymin, current_ymax = ax.get_ylim()
        y_range = current_ymax - current_ymin
        new_ymax = current_ymax + y_range * 0.20
        ax.set_ylim(current_ymin, new_ymax)

        bar_height = current_ymax + y_range * 0.05
        bar_tips = bar_height - y_range * 0.02

        ax.plot([1, 1, 2, 2], [bar_tips, bar_height, bar_height, bar_tips],
               color="black", linewidth=1)
        ax.text(1.5, bar_height + y_range * 0.01, p_text,
               ha="center", va="bottom", fontsize=7)

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    legend_elements = [
        Patch(facecolor="#AED6F1", edgecolor="#1A5276", label="Cognate"),
        Patch(facecolor="#FADBD8", edgecolor="#922B21", label="Non-cognate")
    ]
    fig.legend(handles=legend_elements, loc="lower center",
              ncol=2, fontsize=10, bbox_to_anchor=(0.5, -0.05))

    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {filename}")


def plot_normalised_per_region_boxplots(normalised_per_region_df, region_stats_df, metric="ll"):
    regions = ["FWR1", "CDR1", "FWR2", "CDR2", "FWR3", "CDR3", "FWR4"]

    if metric == "ll":
        cols = [f"{r}_normalised" for r in regions]
        ylabel = "Normalised Mean Log Likelihood per Residue"
        title = "Per-region Normalised Log Likelihood: Cognate vs Non-cognate"
        filename = "per_region_normalised_ll_boxplots.png"
    else:
        cols = [f"{r}_normalised_perplexity" for r in regions]
        ylabel = "Normalised Perplexity"
        title = "Per-region Normalised Perplexity: Cognate vs Non-cognate"
        filename = "per_region_normalised_perplexity_boxplots.png"

    cognate_df = normalised_per_region_df[normalised_per_region_df["cognate"] == True]
    noncognate_df = normalised_per_region_df[normalised_per_region_df["cognate"] == False]

    fig, axes = plt.subplots(1, 7, figsize=(22, 6))
    fig.suptitle(title, fontsize=14, fontweight="bold", y=1.02)

    for idx, (ax, col) in enumerate(zip(axes, cols)):
        cog_vals = cognate_df[col].dropna().values
        noncog_vals = noncognate_df[col].dropna().values

        bp = ax.boxplot([cog_vals, noncog_vals],
                       patch_artist=True,
                       widths=0.4,
                       zorder=2,
                       medianprops=dict(color="black", linewidth=2),
                       whiskerprops=dict(linewidth=1.5),
                       capprops=dict(linewidth=1.5),
                       flierprops=dict(marker="o", markersize=2, alpha=0.3))

        bp["boxes"][0].set_facecolor("#AED6F1")
        bp["boxes"][0].set_edgecolor("#1A5276")
        bp["boxes"][1].set_facecolor("#FADBD8")
        bp["boxes"][1].set_edgecolor("#922B21")

        np.random.seed(42)
        jitter = 0.08
        ax.scatter(np.random.normal(1, jitter, size=len(cog_vals)),
                  cog_vals, alpha=0.3, s=8, color="#1A5276", zorder=3)
        ax.scatter(np.random.normal(2, jitter, size=len(noncog_vals)),
                  noncog_vals, alpha=0.3, s=8, color="#922B21", zorder=3)

        region_label = col.replace("_normalised_perplexity", "").replace("_normalised", "")
        ax.set_title(region_label, fontsize=11, fontweight="bold")
        ax.set_xticks([1, 2])
        ax.set_xticklabels(["Cognate", "Non-\ncognate"], fontsize=8)
        ax.set_ylabel(ylabel if idx == 0 else "", fontsize=9)
        ax.tick_params(axis="y", labelsize=8)

        # P-value annotation — place above current ylim top
        p_adj = region_stats_df.loc[region_stats_df["region"] == col,
                                    "p_adjusted_onetailed"].values[0]

        if p_adj < 0.001:
            p_text = "p<0.001***"
        elif p_adj < 0.01:
            p_text = f"p={p_adj:.3f}**"
        elif p_adj < 0.05:
            p_text = f"p={p_adj:.3f}*"
        else:
            p_text = f"p={p_adj:.3f}\nns"

        # Extend ylim upward first, then place bar in the new space
        current_ymin, current_ymax = ax.get_ylim()
        y_range = current_ymax - current_ymin
        new_ymax = current_ymax + y_range * 0.20
        ax.set_ylim(current_ymin, new_ymax)

        bar_height = current_ymax + y_range * 0.05
        bar_tips = bar_height - y_range * 0.02

        ax.plot([1, 1, 2, 2], [bar_tips, bar_height, bar_height, bar_tips],
               color="black", linewidth=1)
        ax.text(1.5, bar_height + y_range * 0.01, p_text,
               ha="center", va="bottom", fontsize=7)

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    legend_elements = [
        Patch(facecolor="#AED6F1", edgecolor="#1A5276", label="Cognate"),
        Patch(facecolor="#FADBD8", edgecolor="#922B21", label="Non-cognate")
    ]
    fig.legend(handles=legend_elements, loc="lower center",
              ncol=2, fontsize=10, bbox_to_anchor=(0.5, -0.05))

    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {filename}")


def plot_fwr1_ll_deltas(per_region_df):
    df_delta = (
        per_region_df
        .pivot(index=["bispecific", "arm"], columns="cognate", values="FWR1_normalised")
        .reset_index()
        .rename(columns={True: "cognate", False: "noncognate"})
    )

    df_delta["delta"] = df_delta["cognate"] - df_delta["noncognate"]
    values = df_delta["delta"].dropna()

    plt.figure(figsize=(6, 5))
    plt.boxplot(values,
                widths=0.4,
                patch_artist=True,
                zorder=2,
                boxprops=dict(facecolor="#AED6F1", edgecolor="#1A5276"),
                medianprops=dict(color="black", linewidth=2))

    np.random.seed(42)
    jitter = 0.05
    x = np.random.normal(1, jitter, size=len(values))
    plt.scatter(x, values, alpha=0.6, s=20, color="#1A5276", zorder=3)
    plt.axhline(0, linestyle="--", color="black", linewidth=1)
    plt.xticks([1], ["FWR1"])
    plt.ylabel("Normalised Δ Log Likelihood\n(Cognate - Non-cognate)")
    plt.title("FWR1 Normalised Δ Log Likelihood")
    plt.tight_layout()
    plt.savefig("fwr1_delta_normalised_ll.png", dpi=150)
    plt.close()
    print("Saved fwr1_delta_normalised_ll.png")


def main():
    # ── Log likelihoods ──────────────────────────────────────────────────────
    if not os.path.exists("thera_sabdab_cognate_noncognate_log_likelihoods.csv"):
        df_no_common_vl, df_common_vl = load_and_clean_bispecs("TheraSAbDab_020226 1(TheraSAbDab_020226).csv")
        lichen_model = load_model()
        cognate_ll = calculate_log_likelihoods(create_bispecific_pairs(df_no_common_vl, cognate=True), lichen_model)
        noncognate_ll = calculate_log_likelihoods(create_bispecific_pairs(df_no_common_vl, cognate=False), lichen_model)
        all_ll = pd.concat([cognate_ll, noncognate_ll], axis=0).reset_index(drop=True)
        all_ll.to_csv("thera_sabdab_cognate_noncognate_log_likelihoods.csv", index=False)
    else:
        all_ll = pd.read_csv("thera_sabdab_cognate_noncognate_log_likelihoods.csv")

    # ── Normalised log likelihoods ────────────────────────────────────────────
    if not os.path.exists("thera_sabdab_cognate_noncognate_normalised_ll.csv"):
        normalised_ll = normalise_likelihood("thera_sabdab_cognate_noncognate_log_likelihoods.csv")
    else:
        normalised_ll = pd.read_csv("thera_sabdab_cognate_noncognate_normalised_ll.csv")

    # ── Delta normalised log likelihoods ─────────────────────────────────────
    if not os.path.exists("thera_sabdab_bispec_normalised_delta_ll.csv"):
        delta_ll_df = calculate_delta_log_likelihoods(normalised_ll)
        delta_ll_df.to_csv("thera_sabdab_bispec_normalised_delta_ll.csv", index=False)
    else:
        delta_ll_df = pd.read_csv("thera_sabdab_bispec_normalised_delta_ll.csv")

    plot_distribution(delta_ll_df["delta_log_likelihood"],
                     "Delta Normalised Log Likelihood",
                     "Distribution of Delta Normalised Log Likelihoods",
                     "delta_normalised_log_likelihood.png")
    run_analysis(delta_ll_df["delta_log_likelihood"], "log_likelihood")

    # ── Perplexity (derived from normalised log likelihoods) ─────────────────
    if not os.path.exists("thera_sabdab_cognate_noncognate_normalised_perplexity.csv"):
        all_perp = normalised_ll.copy()
        all_perp["perplexity"] = all_perp.apply(
            lambda row: round(math.exp(-row["normalised_ll"] / len(row["light"])), 2),
            axis=1
        )
        all_perp.to_csv("thera_sabdab_cognate_noncognate_normalised_perplexity.csv", index=False)
    else:
        all_perp = pd.read_csv("thera_sabdab_cognate_noncognate_normalised_perplexity.csv")

    if not os.path.exists("thera_sabdab_bispec_normalised_delta_perplexity.csv"):
        delta_perp_df = calculate_delta_perplexity(all_perp)
        delta_perp_df.to_csv("thera_sabdab_bispec_normalised_delta_perplexity.csv", index=False)
    else:
        delta_perp_df = pd.read_csv("thera_sabdab_bispec_normalised_delta_perplexity.csv")

    plot_distribution(delta_perp_df["delta_perplexity"],
                     "Delta Normalised Perplexity",
                     "Distribution of Delta Normalised Perplexity",
                     "delta_normalised_perplexity.png")
    run_analysis(delta_perp_df["delta_perplexity"], "perplexity")

    # ── Per-region log likelihoods ───────────────────────────────────────────
    per_region_path = "thera_sabdab_cognate_noncognate_per_region_ll.csv"
    if not os.path.exists(per_region_path):
        if "df_no_common_vl" not in dir():
            df_no_common_vl, _ = load_and_clean_bispecs("TheraSAbDab_020226 1(TheraSAbDab_020226).csv")
        if "lichen_model" not in dir():
            lichen_model = load_model()
        lichen_model = patch_lichen_model(lichen_model)
        df_cognate_pairs = create_bispecific_pairs(df_no_common_vl, cognate=True)
        df_noncognate_pairs = create_bispecific_pairs(df_no_common_vl, cognate=False)
        all_pairs = pd.concat([df_cognate_pairs, df_noncognate_pairs], axis=0).reset_index(drop=True)
        per_region_ll = light_log_likelihood_per_region(lichen_model, all_pairs)
        per_region_ll.to_csv(per_region_path, index=False)
    else:
        per_region_ll = pd.read_csv(per_region_path)

    # ── Per-region perplexity ────────────────────────────────────────────────
    per_region_perp_path = "thera_sabdab_cognate_noncognate_per_region_perplexity.csv"
    if not os.path.exists(per_region_perp_path):
        per_region_perp = calculate_per_region_perplexity(per_region_ll)
        per_region_perp.to_csv(per_region_perp_path, index=False)
    else:
        per_region_perp = pd.read_csv(per_region_perp_path)

    # Define regions here so it's available for all subsequent blocks
    regions = ["FWR1", "CDR1", "FWR2", "CDR2", "FWR3", "CDR3", "FWR4"]

    # ── Delta per region (log likelihood) ────────────────────────────────────
    delta_per_region_ll_path = "thera_sabdab_delta_per_region_ll.csv"
    if not os.path.exists(delta_per_region_ll_path):
        delta_per_region_ll = calculate_delta_per_region(per_region_ll, regions, label="ll")
        delta_per_region_ll.to_csv(delta_per_region_ll_path, index=False)
    else:
        delta_per_region_ll = pd.read_csv(delta_per_region_ll_path)

    print("\nDelta per region (log likelihood):")
    print(delta_per_region_ll.describe())

    # ── Delta per region (perplexity) ─────────────────────────────────────────
    perplexity_cols = [f"{r}_perplexity" for r in regions]

    delta_per_region_perp_path = "thera_sabdab_delta_per_region_perplexity.csv"
    if not os.path.exists(delta_per_region_perp_path):
        delta_per_region_perp = calculate_delta_per_region(per_region_perp, perplexity_cols, label="perplexity")
        delta_per_region_perp.to_csv(delta_per_region_perp_path, index=False)
    else:
        delta_per_region_perp = pd.read_csv(delta_per_region_perp_path)

    print("\nDelta per region (perplexity):")
    print(delta_per_region_perp.describe())

    # ── Statistical comparison per region ─────────────────────────────────────
    region_stats_ll_path = "per_region_ll_stats.csv"
    if not os.path.exists(region_stats_ll_path):
        region_stats_ll = ttest_per_region_cognate_noncognate(per_region_ll, metric="ll")
        region_stats_ll.to_csv(region_stats_ll_path, index=False)
    else:
        region_stats_ll = pd.read_csv(region_stats_ll_path)

    print("\nPer-region log likelihood comparison (cognate vs non-cognate):")
    print(region_stats_ll.to_string())

    region_stats_perp_path = "per_region_perplexity_stats.csv"
    if not os.path.exists(region_stats_perp_path):
        region_stats_perp = ttest_per_region_cognate_noncognate(per_region_perp, metric="perplexity")
        region_stats_perp.to_csv(region_stats_perp_path, index=False)
    else:
        region_stats_perp = pd.read_csv(region_stats_perp_path)

    print("\nPer-region perplexity comparison (cognate vs non-cognate):")
    print(region_stats_perp.to_string())

    # ── Box-whisker plots ─────────────────────────────────────────────────────
    plot_per_region_boxplots(per_region_ll, region_stats_ll, metric="ll")
    plot_per_region_boxplots(per_region_perp, region_stats_perp, metric="perplexity")

    # ── Normalised per-region log likelihoods ─────────────────────────────────
    normalised_per_region_path = "thera_sabdab_cognate_noncognate_normalised_per_region_ll.csv"
    if not os.path.exists(normalised_per_region_path):
        if "lichen_model" not in dir():
            lichen_model = load_model()
        lichen_model = patch_lichen_model(lichen_model)
        baseline_per_region_path = "thera_sabdab_baseline_per_region_ll.csv"
        if not os.path.exists(baseline_per_region_path):
            baseline_per_region = calculate_baseline_per_region_ll(lichen_model, per_region_ll)
            baseline_per_region.to_csv(baseline_per_region_path, index=False)
        else:
            baseline_per_region = pd.read_csv(baseline_per_region_path)
        normalised_per_region_ll = normalise_per_region_ll(per_region_ll, baseline_per_region)
        normalised_per_region_ll.to_csv(normalised_per_region_path, index=False)
    else:
        normalised_per_region_ll = pd.read_csv(normalised_per_region_path)

    # ── Normalised per-region perplexity ──────────────────────────────────────
    normalised_per_region_perp_path = "thera_sabdab_cognate_noncognate_normalised_per_region_perplexity.csv"
    if not os.path.exists(normalised_per_region_perp_path):
        normalised_per_region_perp = calculate_normalised_per_region_perplexity(normalised_per_region_ll)
        normalised_per_region_perp.to_csv(normalised_per_region_perp_path, index=False)
    else:
        normalised_per_region_perp = pd.read_csv(normalised_per_region_perp_path)

    # ── FWR1 delta plot using normalised values ───────────────────────────────
    plot_fwr1_ll_deltas(normalised_per_region_ll)

    # ── Statistical comparison normalised per region ───────────────────────────
    normalised_region_stats_ll_path = "normalised_per_region_ll_stats.csv"
    if not os.path.exists(normalised_region_stats_ll_path):
        normalised_region_stats_ll = ttest_normalised_per_region(normalised_per_region_ll, metric="ll")
        normalised_region_stats_ll.to_csv(normalised_region_stats_ll_path, index=False)
    else:
        normalised_region_stats_ll = pd.read_csv(normalised_region_stats_ll_path)

    print("\nNormalised per-region log likelihood comparison (cognate vs non-cognate):")
    print(normalised_region_stats_ll.to_string())

    normalised_region_stats_perp_path = "normalised_per_region_perplexity_stats.csv"
    if not os.path.exists(normalised_region_stats_perp_path):
        normalised_region_stats_perp = ttest_normalised_per_region(normalised_per_region_perp, metric="perplexity")
        normalised_region_stats_perp.to_csv(normalised_region_stats_perp_path, index=False)
    else:
        normalised_region_stats_perp = pd.read_csv(normalised_region_stats_perp_path)

    print("\nNormalised per-region perplexity comparison (cognate vs non-cognate):")
    print(normalised_region_stats_perp.to_string())

    # ── Normalised box-whisker plots ───────────────────────────────────────────
    plot_normalised_per_region_boxplots(normalised_per_region_ll, normalised_region_stats_ll, metric="ll")
    plot_normalised_per_region_boxplots(normalised_per_region_perp, normalised_region_stats_perp, metric="perplexity")


if __name__ == "__main__":
    main()
