"""Tests for src/parsing/patient_normalizer.py"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))


class TestExtractAge:
    def test_hyphenated_year_old(self):
        from src.parsing.patient_normalizer import extract_age
        assert extract_age("58-year-old male with") == 58

    def test_yo_abbreviation(self):
        from src.parsing.patient_normalizer import extract_age
        assert extract_age("Patient is a 72yo female") == 72

    def test_age_word(self):
        from src.parsing.patient_normalizer import extract_age
        assert extract_age("age 45 with history of diabetes") == 45

    def test_years_old_spaced(self):
        from src.parsing.patient_normalizer import extract_age
        assert extract_age("A 63 year old woman presented") == 63

    def test_no_age_returns_none(self):
        from src.parsing.patient_normalizer import extract_age
        assert extract_age("Patient with lung cancer and hypertension") is None

    def test_implausible_age_rejected(self):
        from src.parsing.patient_normalizer import extract_age
        assert extract_age("patient 200 years old has") is None

    def test_age_zero_rejected(self):
        from src.parsing.patient_normalizer import extract_age
        assert extract_age("0-year-old infant") is None


class TestExtractGender:
    def test_male_keyword(self):
        from src.parsing.patient_normalizer import extract_gender
        assert extract_gender("A 58-year-old male with hypertension") == "male"

    def test_female_keyword(self):
        from src.parsing.patient_normalizer import extract_gender
        assert extract_gender("58-year-old woman presents with chest pain") == "female"

    def test_she_pronoun(self):
        from src.parsing.patient_normalizer import extract_gender
        assert extract_gender("She was diagnosed with breast cancer last year") == "female"

    def test_he_pronoun(self):
        from src.parsing.patient_normalizer import extract_gender
        assert extract_gender("He has a history of MI and takes aspirin") == "male"

    def test_ambiguous_returns_unknown(self):
        from src.parsing.patient_normalizer import extract_gender
        assert extract_gender("Patient with stage IV lung cancer ECOG 1") == "unknown"

    def test_majority_signal_wins(self):
        """More female signals than male → female."""
        from src.parsing.patient_normalizer import extract_gender
        assert extract_gender("She is a woman, her condition is female cancer") == "female"


class TestExtractEcog:
    def test_ecog_ps(self):
        from src.parsing.patient_normalizer import extract_ecog
        assert extract_ecog("ECOG PS 2") == 2

    def test_ecog_performance_status(self):
        from src.parsing.patient_normalizer import extract_ecog
        assert extract_ecog("ECOG performance status 1") == 1

    def test_ecog_score_of(self):
        from src.parsing.patient_normalizer import extract_ecog
        assert extract_ecog("ECOG score of 0") == 0

    def test_bare_ecog(self):
        from src.parsing.patient_normalizer import extract_ecog
        assert extract_ecog("ECOG 3, tolerating treatment well") == 3

    def test_no_ecog_returns_none(self):
        from src.parsing.patient_normalizer import extract_ecog
        assert extract_ecog("Good functional status, ambulatory") is None


class TestExtractLabValues:
    def test_creatinine(self):
        from src.parsing.patient_normalizer import extract_lab_values
        labs = extract_lab_values("creatinine 1.2 mg/dL on admission")
        assert "creatinine" in labs
        assert "1.2" in labs["creatinine"]

    def test_hemoglobin(self):
        from src.parsing.patient_normalizer import extract_lab_values
        labs = extract_lab_values("hemoglobin 9.5 g/dL, stable")
        assert "hemoglobin" in labs
        assert "9.5" in labs["hemoglobin"]

    def test_alt_ast(self):
        from src.parsing.patient_normalizer import extract_lab_values
        labs = extract_lab_values("ALT 45 U/L, AST 38 U/L")
        assert "alt" in labs
        assert "ast" in labs

    def test_no_labs_returns_empty_dict(self):
        from src.parsing.patient_normalizer import extract_lab_values
        labs = extract_lab_values("Patient with lung cancer, no recent labs")
        assert isinstance(labs, dict)
        assert len(labs) == 0


class TestNormalizePatient:
    def test_empty_string_returns_default_profile(self):
        from src.parsing.patient_normalizer import normalize_patient, PatientProfile
        result = normalize_patient("", use_cache=False)
        assert isinstance(result, PatientProfile)
        assert result.age is None
        assert result.gender == "unknown"
        assert result.conditions == []

    def test_whitespace_only_returns_default(self):
        from src.parsing.patient_normalizer import normalize_patient
        result = normalize_patient("   \n  ", use_cache=False)
        assert result.conditions == []

    def test_deterministic_fields_set_without_llm(self):
        from src.parsing.patient_normalizer import normalize_patient, _LLMExtraction
        note = "58-year-old male, ECOG 1, creatinine 1.1 mg/dL"
        with patch("src.parsing.patient_normalizer._llm_extract", return_value=_LLMExtraction()):
            result = normalize_patient(note, use_cache=False)
        assert result.age == 58
        assert result.gender == "male"
        assert result.ecog_score == 1
        assert "creatinine" in result.lab_values

    def test_llm_fields_integrated(self):
        from src.parsing.patient_normalizer import normalize_patient, _LLMExtraction
        note = "58yo male with NSCLC, taking erlotinib"
        fake = _LLMExtraction(
            conditions=["non-small cell lung cancer"],
            medications=["erlotinib 150mg daily"],
            prior_treatments=["cisplatin chemotherapy"],
            relevant_history=["EGFR exon 19 deletion"],
        )
        with patch("src.parsing.patient_normalizer._llm_extract", return_value=fake):
            result = normalize_patient(note, use_cache=False)
        assert "non-small cell lung cancer" in result.conditions
        assert "erlotinib 150mg daily" in result.medications
        assert "cisplatin chemotherapy" in result.prior_treatments
        assert "EGFR exon 19 deletion" in result.relevant_history

    def test_llm_failure_does_not_crash(self):
        """If LLM extraction raises, normalize_patient should return partial profile."""
        from src.parsing.patient_normalizer import normalize_patient
        note = "72-year-old female with breast cancer"
        with patch("src.parsing.patient_normalizer._llm_extract", side_effect=RuntimeError("api down")):
            result = normalize_patient(note, use_cache=False)
        # Deterministic fields should still be set
        assert result.age == 72
        assert result.gender == "female"
        # LLM fields are empty (safe default)
        assert result.conditions == []

    def test_pydantic_model_validates(self):
        from src.parsing.patient_normalizer import PatientProfile
        p = PatientProfile(age=45, gender="female", conditions=["NSCLC"], ecog_score=1)
        assert p.age == 45
        assert p.ecog_score == 1
        assert p.lab_values == {}

    def test_caching_skips_llm_on_repeat(self):
        """Second call with same note should use cache, not call LLM again."""
        import diskcache, tempfile, os
        from src.parsing.patient_normalizer import normalize_patient, _LLMExtraction

        with tempfile.TemporaryDirectory() as tmpdir:
            cache = diskcache.Cache(tmpdir)
            note = "45-year-old male with AML"
            fake = _LLMExtraction(conditions=["acute myeloid leukemia"])
            call_count = [0]

            def counted_extract(n):
                call_count[0] += 1
                return fake

            with patch("src.parsing.patient_normalizer._llm_extract", side_effect=counted_extract):
                normalize_patient(note, use_cache=True, cache=cache)
                normalize_patient(note, use_cache=True, cache=cache)

        assert call_count[0] == 1, "LLM should only be called once for same note"
