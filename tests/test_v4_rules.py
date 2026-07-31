from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from time_anchor import build_time_anchor
from trial_summary import build_trial_summaries
from validator import normalize_llm_item
from load_rules import TrialRule


class V4RulesTests(unittest.TestCase):
    def test_latest_clinical_event_excludes_create_time(self) -> None:
        anchor = build_time_anchor(
            {
                "emr_document": [{"visit_date": "2026-04-01"}],
                "raw_patient_info": [{"create_time": "2030-01-01"}],
            }
        )
        self.assertEqual(anchor["latest_record_time"], "2026-04-01 00:00:00")
        self.assertEqual(anchor["randomization_enrollment_time"], "2026-04-15 00:00:00")
        self.assertEqual(anchor["treatment_start_time"], "2026-04-17 00:00:00")

    def test_neutral_labels_and_trial_summary(self) -> None:
        rule = TrialRule("R", "T", "1", "", "", "", "入选标准", "年龄≥18岁")
        self.assertEqual(normalize_llm_item({"label": "Y"}, rule)["label"], "满足")
        summary = build_trial_summaries(
            [
                {"患者编号": "P", "试验注册号": "R", "试验标识": "T", "规则标识": "入选标准", "标准编号": "1", "标注结果": "未知"},
                {"患者编号": "P", "试验注册号": "R", "试验标识": "T", "规则标识": "排除标准", "标准编号": "2", "标注结果": "不满足"},
            ]
        )[0]
        self.assertEqual(summary["试验结论"], "符合试验")
        self.assertEqual(summary["是否含未知标准"], "是")


if __name__ == "__main__":
    unittest.main()
