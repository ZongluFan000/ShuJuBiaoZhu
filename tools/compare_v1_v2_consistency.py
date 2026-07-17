from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from pathlib import Path

import pandas as pd


DESKTOP = Path("C:/Users/PC/Desktop/数据标注")
V1_ROOT = DESKTOP / "ShuJuBiaoZhu-clean-sandbox-publish"
V2_ROOT = DESKTOP / "ShuJuBiaoZhu-v2-sandbox"

V1_RESULTS = V1_ROOT / "data/ab_experiment/baseline/runs/20260615_182220/annotation_results.csv"
V2_RESULTS = V2_ROOT / "data/final_v2_120_patients/annotation_results_v2_120_patients_final.csv"
V1_RULES = V1_ROOT / "data/input/rules.xlsx"
V2_RULES = V2_ROOT / "data/input/rules_v2.xlsx"

OUT_DIR = V2_ROOT / "data/final_v2_120_patients/v1_v2_comparison"
OUT_SUMMARY = OUT_DIR / "v1_v2_consistency_summary.json"
OUT_RULE_MAP = OUT_DIR / "v1_v2_rule_mapping.csv"
OUT_PAIRWISE = OUT_DIR / "v1_v2_pairwise_comparable_results.csv"
OUT_XLSX = OUT_DIR / "v1_v2_consistency_report.xlsx"

PATIENT_COL = "患者编号"
TRIAL_COL = "试验标识"
STANDARD_COL = "标准编号"
RULE_TEXT_COL = "标准内容"
LABEL_COL = "标注结果"


def norm_text(value: object) -> str:
    text = "" if pd.isna(value) else str(value)
    text = text.strip().lower()
    text = re.sub(r"\s+", "", text)
    text = re.sub(r"[，。；：、,.;;:()（）【】\[\]{}<>《》\"'“”‘’]", "", text)
    return text


def norm_label(value: object) -> str:
    text = "" if pd.isna(value) else str(value).strip()
    mapping = {
        "符合": "符合",
        "是": "符合",
        "yes": "符合",
        "y": "符合",
        "不符合": "不符合",
        "否": "不符合",
        "no": "不符合",
        "n": "不符合",
        "未知": "未知",
        "无法判断": "未知",
        "不确定": "未知",
        "unknown": "未知",
        "u": "未知",
    }
    return mapping.get(text.lower(), text)


def load_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype=str, encoding="utf-8-sig")


def pick_rule_id(df: pd.DataFrame) -> str:
    for col in ["规则标识", "规则ID", "rule_id", "规则编号"]:
        if col in df.columns:
            return col
    return STANDARD_COL


def rule_col(df: pd.DataFrame, candidates: list[str], fallback_index: int) -> pd.Series:
    for col in candidates:
        if col in df.columns:
            return df[col].fillna("").astype(str)
    if fallback_index < len(df.columns):
        return df.iloc[:, fallback_index].fillna("").astype(str)
    return pd.Series([""] * len(df), index=df.index, dtype=str)


def best_similarity(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    if a == b:
        return 1.0
    if a in b or b in a:
        return min(len(a), len(b)) / max(len(a), len(b))
    return SequenceMatcher(None, a, b).ratio()


def build_mapping(v1_rules: pd.DataFrame, v2_rules: pd.DataFrame) -> pd.DataFrame:
    v1 = v1_rules.copy()
    v2 = v2_rules.copy()
    # Old rules: trial/std/text/type at columns 0/9/10/11.
    v1["_trial"] = rule_col(v1, [TRIAL_COL, "试验编号", "试验ID"], 0)
    v1["_std"] = rule_col(v1, [STANDARD_COL, "标准序号", "规则编号"], 9)
    v1["_text"] = rule_col(v1, [RULE_TEXT_COL, "规则内容", "入排标准内容"], 10)
    v1["_rule_type"] = rule_col(v1, ["规则标识", "规则类型"], 11)
    v1["_norm_text"] = v1["_text"].map(norm_text)

    # V2 rules add a leading note column, so trial/std/text/type shift to 1/10/11/12.
    v2["_trial"] = rule_col(v2, [TRIAL_COL, "试验编号", "试验ID"], 1)
    v2["_std"] = rule_col(v2, [STANDARD_COL, "标准序号", "规则编号"], 10)
    v2["_text"] = rule_col(v2, [RULE_TEXT_COL, "规则内容", "入排标准内容"], 11)
    v2["_rule_type"] = rule_col(v2, ["规则标识", "规则类型"], 12)
    v2["_norm_text"] = v2["_text"].map(norm_text)

    exact_by_trial_std: dict[tuple[str, str], list[int]] = defaultdict(list)
    exact_by_trial_text: dict[tuple[str, str], list[int]] = defaultdict(list)
    trial_groups: dict[str, list[int]] = defaultdict(list)
    for idx, row in v1.iterrows():
        exact_by_trial_std[(row["_trial"], row["_std"])].append(idx)
        exact_by_trial_text[(row["_trial"], row["_norm_text"])].append(idx)
        trial_groups[row["_trial"]].append(idx)

    rows: list[dict[str, object]] = []
    for v2_idx, row in v2.iterrows():
        candidates: list[tuple[str, float, int]] = []
        for idx in exact_by_trial_std.get((row["_trial"], row["_std"]), []):
            sim = best_similarity(row["_norm_text"], v1.at[idx, "_norm_text"])
            candidates.append(("same_trial_standard", sim, idx))
        if not candidates:
            for idx in exact_by_trial_text.get((row["_trial"], row["_norm_text"]), []):
                candidates.append(("same_trial_exact_text", 1.0, idx))
        if not candidates:
            for idx in trial_groups.get(row["_trial"], []):
                sim = best_similarity(row["_norm_text"], v1.at[idx, "_norm_text"])
                if sim >= 0.92:
                    candidates.append(("same_trial_fuzzy_text_0.92", sim, idx))

        if candidates:
            candidates.sort(key=lambda item: (item[1], item[0] == "same_trial_standard"), reverse=True)
            method, score, idx = candidates[0]
            rows.append(
                {
                    "v2_rule_row": int(v2_idx) + 2,
                    "v2_试验标识": row["_trial"],
                    "v2_标准编号": row["_std"],
                    "v2_规则标识": row["_rule_type"],
                    "v2_标准内容": row["_text"],
                    "v1_rule_row": int(idx) + 2,
                    "v1_试验标识": v1.at[idx, "_trial"],
                    "v1_标准编号": v1.at[idx, "_std"],
                    "v1_规则标识": v1.at[idx, "_rule_type"],
                    "v1_标准内容": v1.at[idx, "_text"],
                    "mapping_method": method,
                    "text_similarity": round(score, 4),
                    "mapping_status": "mapped" if score >= 0.92 else "low_confidence",
                }
            )
        else:
            rows.append(
                {
                    "v2_rule_row": int(v2_idx) + 2,
                    "v2_试验标识": row["_trial"],
                    "v2_标准编号": row["_std"],
                    "v2_规则标识": row["_rule_type"],
                    "v2_标准内容": row["_text"],
                    "v1_rule_row": "",
                    "v1_试验标识": "",
                    "v1_标准编号": "",
                    "v1_规则标识": "",
                    "v1_标准内容": "",
                    "mapping_method": "unmapped",
                    "text_similarity": 0.0,
                    "mapping_status": "unmapped",
                }
            )
    return pd.DataFrame(rows)


def compare_results(v1_results: pd.DataFrame, v2_results: pd.DataFrame, mapping: pd.DataFrame) -> pd.DataFrame:
    mapped = mapping[mapping["mapping_status"] == "mapped"].copy()
    v1 = v1_results.copy()
    v2 = v2_results.copy()
    v1["_label_norm"] = v1[LABEL_COL].map(norm_label)
    v2["_label_norm"] = v2[LABEL_COL].map(norm_label)

    v1_lookup: dict[tuple[str, str, str], dict[str, object]] = {}
    for row in v1.to_dict("records"):
        key = (str(row.get(PATIENT_COL, "")), str(row.get(TRIAL_COL, "")), str(row.get(STANDARD_COL, "")))
        v1_lookup.setdefault(key, row)

    records: list[dict[str, object]] = []
    for _, map_row in mapped.iterrows():
        v2_trial = str(map_row["v2_试验标识"])
        v2_std = str(map_row["v2_标准编号"])
        v1_trial = str(map_row["v1_试验标识"])
        v1_std = str(map_row["v1_标准编号"])
        v2_subset = v2[(v2[TRIAL_COL].astype(str) == v2_trial) & (v2[STANDARD_COL].astype(str) == v2_std)]
        for _, v2_row in v2_subset.iterrows():
            patient = str(v2_row[PATIENT_COL])
            key = (patient, v1_trial, v1_std)
            hit = v1_lookup.get(key)
            if hit is None:
                records.append(
                    {
                        PATIENT_COL: patient,
                        "v2_试验标识": v2_trial,
                        "v2_标准编号": v2_std,
                        "v1_试验标识": v1_trial,
                        "v1_标准编号": v1_std,
                        "v2_标注结果": v2_row[LABEL_COL],
                        "v1_标注结果": "",
                        "v2_label_norm": v2_row["_label_norm"],
                        "v1_label_norm": "",
                        "一致": False,
                        "missing_v1": True,
                        "mapping_method": map_row["mapping_method"],
                        "text_similarity": map_row["text_similarity"],
                    }
                )
                continue
            records.append(
                {
                    PATIENT_COL: patient,
                    "v2_试验标识": v2_trial,
                    "v2_标准编号": v2_std,
                    "v1_试验标识": v1_trial,
                    "v1_标准编号": v1_std,
                    "v2_标注结果": v2_row[LABEL_COL],
                    "v1_标注结果": hit[LABEL_COL],
                    "v2_label_norm": v2_row["_label_norm"],
                    "v1_label_norm": hit["_label_norm"],
                    "一致": v2_row["_label_norm"] == hit["_label_norm"],
                    "missing_v1": False,
                    "mapping_method": map_row["mapping_method"],
                    "text_similarity": map_row["text_similarity"],
                }
            )
    return pd.DataFrame(records)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    v1_results = load_csv(V1_RESULTS)
    v2_results = load_csv(V2_RESULTS)
    v1_rules = v1_results[[TRIAL_COL, STANDARD_COL, "规则标识", RULE_TEXT_COL]].drop_duplicates(
        subset=[TRIAL_COL, STANDARD_COL]
    )
    v2_rules = v2_results[[TRIAL_COL, STANDARD_COL, "规则标识", RULE_TEXT_COL]].drop_duplicates(
        subset=[TRIAL_COL, STANDARD_COL]
    )

    mapping = build_mapping(v1_rules, v2_rules)
    pairwise = compare_results(v1_results, v2_results, mapping)

    mapped = mapping[mapping["mapping_status"] == "mapped"]
    comparable = pairwise[pairwise["missing_v1"] == False].copy()
    method_summary = (
        comparable.groupby("mapping_method")["一致"]
        .agg(["count", "sum", "mean"])
        .reset_index()
        .rename(columns={"count": "可比结果数", "sum": "一致数", "mean": "一致率"})
    )
    label_matrix = pd.crosstab(comparable["v1_label_norm"], comparable["v2_label_norm"], dropna=False)
    by_rule = (
        comparable.groupby(["v2_试验标识", "v2_标准编号", "v1_试验标识", "v1_标准编号"])["一致"]
        .agg(["count", "sum", "mean"])
        .reset_index()
        .rename(columns={"count": "患者数", "sum": "一致数", "mean": "一致率"})
    )
    by_patient = (
        comparable.groupby(PATIENT_COL)["一致"]
        .agg(["count", "sum", "mean"])
        .reset_index()
        .rename(columns={"count": "可比规则数", "sum": "一致数", "mean": "一致率"})
    )

    summary = {
        "v1_results": str(V1_RESULTS),
        "v2_results": str(V2_RESULTS),
        "v1_rows": int(len(v1_results)),
        "v2_rows": int(len(v2_results)),
        "v1_patients": int(v1_results[PATIENT_COL].nunique()),
        "v2_patients": int(v2_results[PATIENT_COL].nunique()),
        "v1_unique_rules_by_trial_standard": int(v1_results[[TRIAL_COL, STANDARD_COL]].drop_duplicates().shape[0]),
        "v2_unique_rules_by_trial_standard": int(v2_results[[TRIAL_COL, STANDARD_COL]].drop_duplicates().shape[0]),
        "v2_rules_total": int(v2_results[[TRIAL_COL, STANDARD_COL]].drop_duplicates().shape[0]),
        "v2_rules_mapped": int(len(mapped)),
        "v2_rules_unmapped": int((mapping["mapping_status"] != "mapped").sum()),
        "comparable_result_rows": int(len(comparable)),
        "overall_consistent_rows": int(comparable["一致"].sum()) if len(comparable) else 0,
        "overall_consistency_rate": float(comparable["一致"].mean()) if len(comparable) else None,
        "mapping_methods": Counter(mapping["mapping_method"]).most_common(),
        "label_matrix": label_matrix.to_dict(),
    }

    mapping.to_csv(OUT_RULE_MAP, index=False, encoding="utf-8-sig")
    pairwise.to_csv(OUT_PAIRWISE, index=False, encoding="utf-8-sig")
    with OUT_SUMMARY.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    with pd.ExcelWriter(OUT_XLSX, engine="openpyxl") as writer:
        pd.DataFrame([summary]).to_excel(writer, index=False, sheet_name="summary")
        method_summary.to_excel(writer, index=False, sheet_name="by_mapping_method")
        label_matrix.reset_index().to_excel(writer, index=False, sheet_name="label_matrix")
        by_rule.sort_values("一致率").to_excel(writer, index=False, sheet_name="by_rule")
        by_patient.sort_values("一致率").to_excel(writer, index=False, sheet_name="by_patient")
        mapping.to_excel(writer, index=False, sheet_name="rule_mapping")
        pairwise[pairwise["一致"] == False].head(5000).to_excel(writer, index=False, sheet_name="diff_examples")

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
