"""
Step 1 — Extract and prepare labels from the Galaxy Zoo 2 catalogue.

Reads the raw Galaxy Zoo 2 catalogue (Hart et al. 2016) and produces a
clean CSV with:
  * Galaxy ID, RA, Dec
  * Debiased voting probabilities for Q1 and Q2
  * Hard (argmax) integer labels ready for training

Dataset source: https://data.galaxyzoo.org/
file downloaded: https://gz2hart.s3.amazonaws.com/gz2_hart16.csv.gz

Paper: Willett et al. 2013, MNRAS, 435, 2835
       https://academic.oup.com/mnras/article/435/4/2835/1022913
"""

import pandas as pd

from config import RAW_CATALOG_PATH, LABELS_PATH

RAW_COLUMNS = [
    "dr7objid",
    "ra", "dec",
    "t01_smooth_or_features_a01_smooth_debiased",
    "t01_smooth_or_features_a02_features_or_disk_debiased",
    "t01_smooth_or_features_a03_star_or_artifact_debiased",
    "t02_edgeon_a04_yes_debiased",
    "t02_edgeon_a05_no_debiased",
]

RENAME_MAP = {
    "dr7objid":                                              "id",
    "t01_smooth_or_features_a01_smooth_debiased":            "prob_smooth",
    "t01_smooth_or_features_a02_features_or_disk_debiased":  "prob_features_or_disk",
    "t01_smooth_or_features_a03_star_or_artifact_debiased":  "prob_star_or_artifact",
    "t02_edgeon_a04_yes_debiased":                           "prob_edge_on",
    "t02_edgeon_a05_no_debiased":                            "prob_not_edge_on",
}

def main():

    df = pd.read_csv(
        RAW_CATALOG_PATH,
        usecols=RAW_COLUMNS,
        dtype={"dr7objid": "string"},
    )
    df = df.rename(columns=RENAME_MAP)

    q1_cols = ["prob_smooth", "prob_features_or_disk", "prob_star_or_artifact"]
    df["q1_label_name"] = df[q1_cols].idxmax(axis=1)
    q1_map = {"prob_smooth": 0, "prob_features_or_disk": 1, "prob_star_or_artifact": 2}
    df["q1_label"] = df["q1_label_name"].map(q1_map)

    q2_cols = ["prob_edge_on", "prob_not_edge_on"]
    df["q2_label_name"] = df[q2_cols].idxmax(axis=1)
    q2_map = {"prob_edge_on": 0, "prob_not_edge_on": 1}
    df["q2_label"] = df["q2_label_name"].map(q2_map)

    df.to_csv(LABELS_PATH, index=False)

    print(f"Saved labels for Q1 + Q2 → {LABELS_PATH}")
    print(f"  Total galaxies : {len(df):,}")
    print(f"  Q1 distribution: {df['q1_label'].value_counts().to_dict()}")
    print(f"  Q2 distribution: {df['q2_label'].value_counts().to_dict()}")


if __name__ == "__main__":
    main()