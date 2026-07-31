from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from load_rules import TrialRule, load_rules
from evidence_builder import EvidencePackage
from prompting import build_optimized_batch_prompt
from rule_optimizer import (
    build_judgment_units,
    build_optimized_batches,
    effective_batch_rule_limit,
    rule_complexity_cost,
    unit_similarity,
    validate_optimization_plan,
)
from validator import normalize_llm_batch_result_strict


class RuleOptimizerTests(unittest.TestCase):
    def test_all_original_rules_are_preserved_as_members(self) -> None:
        rules = load_rules(ROOT / "data" / "input" / "rules_v3_standard.xlsx")
        units = build_judgment_units(
            rules,
            merge_exact_duplicates=True,
            reviewed_groups_path=ROOT / "config" / "reviewed_equivalent_rules.json",
        )
        members = [rule for unit in units for rule in unit.members]
        original_keys = {(rule.trial_id, rule.standard_no) for rule in rules}
        member_keys = {(rule.trial_id, rule.standard_no) for rule in members}
        self.assertEqual(len(rules), len(members))
        self.assertEqual(original_keys, member_keys)

    def test_context_dependent_duplicates_are_not_merged(self) -> None:
        rules = [
            self.make_rule("试验_1", "1", "研究给药开始前4周内不得接种活疫苗。"),
            self.make_rule("试验_2", "1", "研究给药开始前4周内不得接种活疫苗。"),
        ]
        units = build_judgment_units(rules, merge_exact_duplicates=True)
        self.assertEqual(2, len(units))
        self.assertTrue(all(len(unit.members) == 1 for unit in units))

    def test_context_free_exact_duplicates_are_merged(self) -> None:
        rules = [
            self.make_rule("试验_1", "1", "预期生存期≥12周。", "入选标准"),
            self.make_rule("试验_2", "1", "预期生存期≥12周。", "入选标准"),
        ]
        units = build_judgment_units(rules, merge_exact_duplicates=True)
        self.assertEqual(1, len(units))
        self.assertEqual(2, len(units[0].members))

    def test_reviewed_groups_require_explicit_enablement(self) -> None:
        rules = [
            self.make_rule("trial_1", "1", "age >= 18"),
            self.make_rule("trial_2", "2", "age at least 18"),
        ]
        registry = ROOT / "data" / "test_reviewed_groups.json"
        registry.parent.mkdir(parents=True, exist_ok=True)
        registry.write_text(
            '{"groups":[{"review_status":"approved","members":['
            '{"trial_id":"trial_1","standard_no":"1"},'
            '{"trial_id":"trial_2","standard_no":"2"}]}]}',
            encoding="utf-8",
        )
        try:
            units = build_judgment_units(
                rules,
                merge_exact_duplicates=False,
                reviewed_groups_path=registry,
                use_reviewed_equivalent_groups=False,
            )
            self.assertEqual(2, len(units))
            self.assertTrue(all(unit.merge_type == "none" for unit in units))
        finally:
            registry.unlink(missing_ok=True)

    def test_optimized_batches_keep_complex_batches_small(self) -> None:
        rules = [
            self.make_rule(f"试验_{idx}", "1", f"患者既往接受治疗且出现进展，但特殊情况除外，规则编号{idx}。")
            for idx in range(8)
        ]
        units = build_judgment_units(rules, merge_exact_duplicates=True)
        batches = build_optimized_batches(units, max_rules=10, max_rule_chars=3000, max_complex_rules=3)
        self.assertLessEqual(max(map(len, batches)), 3)

    def test_strict_batch_validation_rejects_missing_rule(self) -> None:
        rules = [
            self.make_rule("试验_1", "1", "年龄≥18岁。"),
            self.make_rule("试验_2", "2", "性别不限。"),
        ]
        payload = {
            "items": [
                {
                    "trial_id": "试验_1",
                    "standard_no": "1",
                    "label": "符合",
                }
            ]
        }
        with self.assertRaises(ValueError):
            normalize_llm_batch_result_strict(payload, rules)

    def test_output_budget_caps_batch_size(self) -> None:
        self.assertEqual(
            8,
            effective_batch_rule_limit(
                configured_max_rules=10,
                model_max_tokens=1024,
                estimated_output_tokens_per_rule=110,
                output_token_reserve=128,
            ),
        )

    def test_strict_batch_validation_rejects_invalid_label(self) -> None:
        rules = [self.make_rule("trial_1", "1", "age >= 18")]
        payload = {
            "items": [
                {
                    "trial_id": "trial_1",
                    "standard_no": "1",
                    "label": "possibly_eligible",
                }
            ]
        }
        with self.assertRaises(ValueError):
            normalize_llm_batch_result_strict(payload, rules)

    def test_plan_validation_rejects_missing_unit(self) -> None:
        rules = [
            self.make_rule("trial_1", "1", "age >= 18"),
            self.make_rule("trial_2", "2", "female only"),
        ]
        units = build_judgment_units(rules, merge_exact_duplicates=False)
        with self.assertRaises(ValueError):
            validate_optimization_plan(rules, units, [[units[0]]])

    def test_optimized_prompt_uses_compact_schema_without_conflicting_full_schema(self) -> None:
        rule = self.make_rule("试验_1", "1", "年龄≥18岁。", "入选标准")
        evidence = EvidencePackage(patient_sn="P1", text="birth_date=1980-01-01", sources=["raw_patient_info"])
        prompt = build_optimized_batch_prompt(
            ROOT / "config" / "prompt_templates" / "label_rules.md",
            "P1",
            [(rule, evidence)],
        )
        compact_section = prompt.split("严格独立判断要求：", 1)[1]
        self.assertIn('{"i":[{"n":1,"l":"Y/N/U"', compact_section)
        self.assertIn("只能用R号对齐结果", compact_section)
        self.assertNotIn('"trial_register_id": "..."', prompt)

    def test_similarity_aware_batching_places_related_rules_together(self) -> None:
        rules = [
            self.make_rule("试验_1", "1", "既往存在严重心血管疾病史。"),
            self.make_rule("试验_2", "1", "既往存在心脏疾病史。"),
            self.make_rule("试验_3", "1", "既往存在活动性乙型肝炎感染。"),
            self.make_rule("试验_4", "1", "既往存在丙型肝炎感染。"),
        ]
        units = build_judgment_units(rules, merge_exact_duplicates=False)
        batches = build_optimized_batches(units, max_rules=2, max_rule_chars=3000, max_complex_rules=2)
        pairs = [
            {unit.representative.trial_id for unit in batch}
            for batch in batches
        ]
        self.assertIn({"试验_1", "试验_2"}, pairs)
        self.assertIn({"试验_3", "试验_4"}, pairs)
        self.assertGreater(unit_similarity(units[0], units[1]), unit_similarity(units[0], units[2]))

    def test_complexity_budget_shrinks_medium_batches(self) -> None:
        rules = [
            self.make_rule(
                f"试验_{idx}",
                "1",
                f"既往接受化疗且在首次给药前4周内出现影像学进展，规则{idx}。",
            )
            for idx in range(8)
        ]
        units = build_judgment_units(rules, merge_exact_duplicates=False)
        batches = build_optimized_batches(
            units,
            max_rules=10,
            max_rule_chars=3000,
            max_complex_rules=3,
            max_simple_rules=10,
            max_normal_rules=9,
            complexity_budget=32,
        )
        self.assertTrue(
            all(sum(rule_complexity_cost(unit) for unit in batch) <= 32 or len(batch) == 1 for batch in batches)
        )
        self.assertEqual(
            10,
            effective_batch_rule_limit(
                configured_max_rules=10,
                model_max_tokens=2048,
                estimated_output_tokens_per_rule=110,
                output_token_reserve=128,
            ),
        )

    @staticmethod
    def make_rule(trial_id: str, standard_no: str, text: str, rule_type: str = "排除标准") -> TrialRule:
        return TrialRule(
            trial_id=trial_id,
            trial_register_id=f"CTR-{trial_id}",
            protocol_no="P",
            cancer_type="",
            cancer_category="",
            standard_no=standard_no,
            rule_text=text,
            rule_type=rule_type,
        )


if __name__ == "__main__":
    unittest.main()
