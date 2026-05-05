"""Tests for src/parsing/criteria_parser.py"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

# ---------------------------------------------------------------------------
# Section splitting helpers
# ---------------------------------------------------------------------------

class TestSplitSections:
    def test_standard_headers(self):
        from src.parsing.criteria_parser import _split_sections
        text = "Inclusion Criteria:\n- item1\n\nExclusion Criteria:\n- item2"
        inc, exc = _split_sections(text)
        assert "item1" in inc
        assert "item2" in exc

    def test_case_insensitive_headers(self):
        from src.parsing.criteria_parser import _split_sections
        text = "INCLUSION CRITERIA:\n- A\nEXCLUSION CRITERIA:\n- B"
        inc, exc = _split_sections(text)
        assert "A" in inc and "B" in exc

    def test_exclusion_before_inclusion(self):
        from src.parsing.criteria_parser import _split_sections
        text = "Exclusion Criteria:\n- excl\nInclusion Criteria:\n- incl"
        inc, exc = _split_sections(text)
        assert "incl" in inc
        assert "excl" in exc

    def test_only_inclusion_header(self):
        from src.parsing.criteria_parser import _split_sections
        text = "Inclusion Criteria:\n- only item"
        inc, exc = _split_sections(text)
        assert "only item" in inc
        assert exc == ""

    def test_no_headers_defaults_to_inclusion(self):
        from src.parsing.criteria_parser import _split_sections
        text = "Patient must be 18+ years old"
        inc, exc = _split_sections(text)
        assert "18+" in inc
        assert exc == ""

    def test_empty_string_returns_empty_pair(self):
        from src.parsing.criteria_parser import _split_sections
        inc, exc = _split_sections("")
        assert inc == "" and exc == ""


class TestExtractItems:
    def test_dash_bullets(self):
        from src.parsing.criteria_parser import _extract_items
        text = "- Age 18+\n- Diagnosis of cancer\n- ECOG 0-2"
        items = _extract_items(text)
        assert len(items) == 3
        assert items[0] == "Age 18+"

    def test_numbered_list(self):
        from src.parsing.criteria_parser import _extract_items
        text = "1. First criterion\n2. Second criterion"
        items = _extract_items(text)
        assert len(items) == 2

    def test_multiline_item_joined(self):
        from src.parsing.criteria_parser import _extract_items
        text = "- First line of\n  a long criterion\n- Next criterion"
        items = _extract_items(text)
        assert len(items) == 2
        assert "First line" in items[0] and "long criterion" in items[0]

    def test_empty_section_returns_empty(self):
        from src.parsing.criteria_parser import _extract_items
        assert _extract_items("") == []
        assert _extract_items("   \n  ") == []

    def test_no_bullets_treats_each_line_as_item(self):
        from src.parsing.criteria_parser import _extract_items
        text = "Age 18+\nNo prior therapy"
        items = _extract_items(text)
        assert len(items) == 2


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------

class TestNormalize:
    def test_operator_unicode_normalised(self):
        from src.parsing.criteria_parser import _normalize, Criterion
        c = Criterion(text="Creatinine ≤ 1.5", type="inclusion",
                      category="lab_value", deterministic=True,
                      numeric_threshold=1.5, numeric_unit="mg/dL", numeric_operator="≤")
        result = _normalize(c)
        assert result.numeric_operator == "<="

    def test_invalid_category_defaults_to_other(self):
        from src.parsing.criteria_parser import _normalize, Criterion
        c = Criterion(text="Some criterion", type="inclusion",
                      category="unknown_cat", deterministic=False)
        result = _normalize(c)
        assert result.category == "other"

    def test_valid_category_preserved(self):
        from src.parsing.criteria_parser import _normalize, Criterion
        for cat in ("age", "gender", "lab_value", "prior_treatment", "diagnosis", "other"):
            c = Criterion(text="x", type="inclusion", category=cat, deterministic=False)
            assert _normalize(c).category == cat

    def test_deterministic_downgraded_when_no_threshold(self):
        """age/lab_value criterion marked deterministic but has no numeric_threshold → downgrade."""
        from src.parsing.criteria_parser import _normalize, Criterion
        c = Criterion(text="Adequate renal function", type="inclusion",
                      category="lab_value", deterministic=True,
                      numeric_threshold=None)
        result = _normalize(c)
        assert result.deterministic is False

    def test_deterministic_preserved_with_threshold(self):
        from src.parsing.criteria_parser import _normalize, Criterion
        c = Criterion(text="Age >= 18", type="inclusion",
                      category="age", deterministic=True,
                      numeric_threshold=18.0, numeric_operator=">=")
        result = _normalize(c)
        assert result.deterministic is True


# ---------------------------------------------------------------------------
# Type fixing
# ---------------------------------------------------------------------------

class TestFixTypes:
    def test_wrong_type_corrected_from_exclusion_list(self):
        from src.parsing.criteria_parser import _fix_types, Criterion
        c = Criterion(text="Prior platinum chemotherapy", type="inclusion",
                      category="prior_treatment", deterministic=False)
        exc_items = ["Prior platinum chemotherapy"]
        result = _fix_types([c], inc_items=[], exc_items=exc_items)
        assert result[0].type == "exclusion"

    def test_correct_type_unchanged(self):
        from src.parsing.criteria_parser import _fix_types, Criterion
        c = Criterion(text="Age 18+", type="inclusion",
                      category="age", deterministic=True,
                      numeric_threshold=18.0)
        result = _fix_types([c], inc_items=["Age 18+"], exc_items=[])
        assert result[0].type == "inclusion"

    def test_unknown_text_type_unchanged(self):
        """If text matches neither list, trust LLM's type."""
        from src.parsing.criteria_parser import _fix_types, Criterion
        c = Criterion(text="Something novel", type="exclusion",
                      category="other", deterministic=False)
        result = _fix_types([c], inc_items=["Different text"], exc_items=["Another text"])
        assert result[0].type == "exclusion"


# ---------------------------------------------------------------------------
# parse_criteria — end-to-end (mocked LLM)
# ---------------------------------------------------------------------------

class TestParseCriteria:
    def _make_criterion(self, text: str, ctype: str, cat: str = "other",
                        det: bool = False) -> "Criterion":
        from src.parsing.criteria_parser import Criterion
        return Criterion(text=text, type=ctype, category=cat, deterministic=det)

    def test_empty_input_returns_empty(self):
        from src.parsing.criteria_parser import parse_criteria
        result = parse_criteria("", use_cache=False)
        assert result.criteria == []

    def test_whitespace_only_returns_empty(self):
        from src.parsing.criteria_parser import parse_criteria
        result = parse_criteria("   \n\n   ", use_cache=False)
        assert result.criteria == []

    def test_llm_result_returned(self):
        from src.parsing.criteria_parser import parse_criteria, _CriteriaExtraction, Criterion
        fake_criteria = [
            Criterion(text="Age >= 18", type="inclusion", category="age",
                      deterministic=True, numeric_threshold=18.0, numeric_operator=">="),
            Criterion(text="Prior cisplatin therapy", type="exclusion",
                      category="prior_treatment", deterministic=False),
        ]
        fake_extraction = _CriteriaExtraction(criteria=fake_criteria)

        text = "Inclusion Criteria:\n- Age >= 18\n\nExclusion Criteria:\n- Prior cisplatin therapy"
        with patch("src.parsing.criteria_parser.complete_structured", return_value=fake_extraction):
            result = parse_criteria(text, use_cache=False)

        assert len(result.criteria) == 2
        age_crit = next(c for c in result.criteria if c.category == "age")
        assert age_crit.deterministic is True
        assert age_crit.numeric_threshold == 18.0

    def test_llm_failure_returns_fallback(self):
        """If LLM fails twice, returns raw-text fallback (category='other')."""
        from src.parsing.criteria_parser import parse_criteria
        text = "Inclusion Criteria:\n- Criterion 1\n\nExclusion Criteria:\n- Criterion 2"
        with patch("src.parsing.criteria_parser.complete_structured",
                   side_effect=RuntimeError("LLM down")):
            result = parse_criteria(text, use_cache=False)
        # Fallback should produce one criterion per item
        assert len(result.criteria) >= 2
        assert all(c.category == "other" for c in result.criteria)

    def test_cache_hit_skips_llm(self):
        import diskcache, tempfile
        from src.parsing.criteria_parser import parse_criteria, ParsedCriteria, Criterion

        text = "Inclusion Criteria:\n- Age >= 18"
        fake_result = ParsedCriteria(criteria=[
            Criterion(text="Age >= 18", type="inclusion", category="age", deterministic=True,
                      numeric_threshold=18.0, numeric_operator=">=")
        ])
        call_count = [0]

        def counted_complete(*args, **kwargs):
            call_count[0] += 1
            from src.parsing.criteria_parser import _CriteriaExtraction
            return _CriteriaExtraction(criteria=fake_result.criteria)

        with tempfile.TemporaryDirectory() as tmpdir:
            cache = diskcache.Cache(tmpdir)
            with patch("src.parsing.criteria_parser.complete_structured",
                       side_effect=counted_complete):
                parse_criteria(text, use_cache=True, cache=cache)
                parse_criteria(text, use_cache=True, cache=cache)

        assert call_count[0] == 1, "LLM must not be called twice for same criteria text"

    def test_cache_stores_json_string(self):
        """Verify cache stores raw JSON string, not dict or Pydantic object."""
        import diskcache, tempfile
        from src.parsing.criteria_parser import parse_criteria, ParsedCriteria, Criterion

        text = "Inclusion Criteria:\n- ECOG 0-1"
        fake = Criterion(text="ECOG 0-1", type="inclusion", category="lab_value",
                         deterministic=True, numeric_threshold=1.0, numeric_operator="<=")

        with tempfile.TemporaryDirectory() as tmpdir:
            cache = diskcache.Cache(tmpdir)
            with patch("src.parsing.criteria_parser.complete_structured") as mock_llm:
                from src.parsing.criteria_parser import _CriteriaExtraction
                mock_llm.return_value = _CriteriaExtraction(criteria=[fake])
                parse_criteria(text, use_cache=True, cache=cache)

            # Verify the stored value is a JSON string
            import hashlib
            key = hashlib.md5(text.encode()).hexdigest()
            stored = cache.get(key)
            assert isinstance(stored, str), "Cache must store JSON string, not dict or model"
            assert stored.startswith("{"), "Stored value must be JSON object"

    def test_inclusion_exclusion_types_correct(self):
        """Verify type field matches section for each criterion."""
        from src.parsing.criteria_parser import parse_criteria, _CriteriaExtraction, Criterion
        inc_crit = Criterion(text="Age 18+", type="inclusion", category="age", deterministic=True,
                             numeric_threshold=18.0, numeric_operator=">=")
        exc_crit = Criterion(text="Prior therapy", type="exclusion", category="prior_treatment",
                             deterministic=False)
        fake = _CriteriaExtraction(criteria=[inc_crit, exc_crit])
        text = "Inclusion Criteria:\n- Age 18+\n\nExclusion Criteria:\n- Prior therapy"
        with patch("src.parsing.criteria_parser.complete_structured", return_value=fake):
            result = parse_criteria(text, use_cache=False)
        types = {c.text: c.type for c in result.criteria}
        assert types.get("Age 18+") == "inclusion"
        assert types.get("Prior therapy") == "exclusion"

    def test_batching_triggered_for_large_criteria(self):
        """Criteria with > 20 items should trigger batching (logged)."""
        from src.parsing.criteria_parser import parse_criteria, _CriteriaExtraction, Criterion

        # Build a criteria text with 15 inclusion + 15 exclusion items
        inc_lines = "\n".join(f"- Inclusion item {i}" for i in range(15))
        exc_lines = "\n".join(f"- Exclusion item {i}" for i in range(15))
        text = f"Inclusion Criteria:\n{inc_lines}\n\nExclusion Criteria:\n{exc_lines}"

        call_count = [0]
        def batch_complete(*args, **kwargs):
            call_count[0] += 1
            return _CriteriaExtraction(criteria=[])

        with patch("src.parsing.criteria_parser.complete_structured",
                   side_effect=batch_complete):
            result = parse_criteria(text, use_cache=False)

        # Batching means > 1 LLM call
        assert call_count[0] > 1, "Large criteria should trigger multiple LLM batch calls"

    def test_pydantic_model_fields(self):
        from src.parsing.criteria_parser import Criterion, ParsedCriteria
        c = Criterion(text="test", type="inclusion", category="age",
                      deterministic=True, numeric_threshold=18.0,
                      numeric_unit="years", numeric_operator=">=")
        p = ParsedCriteria(criteria=[c])
        assert p.criteria[0].numeric_threshold == 18.0
        assert p.criteria[0].numeric_operator == ">="
