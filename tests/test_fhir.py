"""Unit tests for aegis.fhir — FHIR Bundle loading and resource lookups."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from aegis.fhir import FHIRStore

SAMPLE_FILE = Path("data/synthea/sample_patient_joao.json")
PATIENT_ID = "patient-joao-001"


@pytest.fixture
def store() -> FHIRStore:
    """A FHIRStore pre-loaded with the sample patient bundle."""
    s = FHIRStore()
    s.load_bundle(SAMPLE_FILE)
    return s


# ------------------------------------------------------------------
# Loading
# ------------------------------------------------------------------


class TestLoadBundle:
    """Verify that load_bundle correctly parses a FHIR Bundle."""

    def test_load_registers_patient(self, store: FHIRStore):
        assert store.get_patient(PATIENT_ID) is not None

    def test_load_indexes_conditions(self, store: FHIRStore):
        assert len(store.get_conditions(PATIENT_ID)) == 3

    def test_load_indexes_medications(self, store: FHIRStore):
        assert len(store.get_medications(PATIENT_ID)) == 3

    def test_load_indexes_observations(self, store: FHIRStore):
        assert len(store.get_observations(PATIENT_ID)) == 4

    def test_load_rejects_non_bundle(self, tmp_path: Path):
        bad = tmp_path / "bad.json"
        bad.write_text(json.dumps({"resourceType": "Patient", "id": "x"}))
        s = FHIRStore()
        with pytest.raises(ValueError, match="Not a FHIR Bundle"):
            s.load_bundle(bad)

    def test_load_handles_empty_bundle(self, tmp_path: Path):
        empty = tmp_path / "empty.json"
        empty.write_text(json.dumps({"resourceType": "Bundle", "type": "collection", "entry": []}))
        s = FHIRStore()
        s.load_bundle(empty)
        assert s.list_patients() == []


# ------------------------------------------------------------------
# Patient lookups
# ------------------------------------------------------------------


class TestListPatients:
    """Verify list_patients returns correct summaries."""

    def test_returns_one_patient(self, store: FHIRStore):
        patients = store.list_patients()
        assert len(patients) == 1

    def test_patient_has_id_and_name(self, store: FHIRStore):
        patient = store.list_patients()[0]
        assert patient["id"] == PATIENT_ID
        assert "João" in patient["name"]
        assert "Silva" in patient["name"]

    def test_empty_store_returns_empty_list(self):
        s = FHIRStore()
        assert s.list_patients() == []


class TestGetPatient:
    """Verify get_patient returns the full Patient resource."""

    def test_returns_patient_resource(self, store: FHIRStore):
        patient = store.get_patient(PATIENT_ID)
        assert patient is not None
        assert patient["resourceType"] == "Patient"
        assert patient["gender"] == "male"
        assert patient["birthDate"] == "1960-03-15"

    def test_returns_none_for_unknown_id(self, store: FHIRStore):
        assert store.get_patient("nonexistent") is None


# ------------------------------------------------------------------
# Resource lookups
# ------------------------------------------------------------------


class TestGetConditions:
    """Verify condition lookups return expected data."""

    def test_condition_count(self, store: FHIRStore):
        conditions = store.get_conditions(PATIENT_ID)
        assert len(conditions) == 3

    def test_condition_has_code_and_text(self, store: FHIRStore):
        conditions = store.get_conditions(PATIENT_ID)
        texts = [c["code"]["text"] for c in conditions]
        assert "Hipertensão arterial sistêmica" in texts
        assert "Diabetes mellitus tipo 2" in texts
        assert "Insuficiência cardíaca congestiva" in texts

    def test_empty_for_unknown_patient(self, store: FHIRStore):
        assert store.get_conditions("nonexistent") == []


class TestGetMedications:
    """Verify medication lookups return expected data."""

    def test_medication_count(self, store: FHIRStore):
        meds = store.get_medications(PATIENT_ID)
        assert len(meds) == 3

    def test_medication_has_name(self, store: FHIRStore):
        meds = store.get_medications(PATIENT_ID)
        texts = [m["medicationCodeableConcept"]["text"] for m in meds]
        assert "Losartana 50mg" in texts
        assert "Metformina 850mg" in texts

    def test_empty_for_unknown_patient(self, store: FHIRStore):
        assert store.get_medications("nonexistent") == []


class TestGetObservations:
    """Verify observation (vital signs) lookups return expected data."""

    def test_observation_count(self, store: FHIRStore):
        obs = store.get_observations(PATIENT_ID)
        assert len(obs) == 4

    def test_observation_has_vital_sign(self, store: FHIRStore):
        obs = store.get_observations(PATIENT_ID)
        texts = [o["code"]["text"] for o in obs]
        assert "Pressão arterial" in texts
        assert "Frequência cardíaca" in texts
        assert "Peso" in texts
        assert "Altura" in texts

    def test_empty_for_unknown_patient(self, store: FHIRStore):
        assert store.get_observations("nonexistent") == []


# ------------------------------------------------------------------
# Generic resource lookup
# ------------------------------------------------------------------


class TestGetResources:
    """Verify the generic get_resources method."""

    def test_get_by_resource_type(self, store: FHIRStore):
        conditions = store.get_resources(PATIENT_ID, "Condition")
        assert len(conditions) == 3

    def test_unknown_resource_type_returns_empty(self, store: FHIRStore):
        assert store.get_resources(PATIENT_ID, "Procedure") == []


# ------------------------------------------------------------------
# Synthea transaction bundle format
# ------------------------------------------------------------------


class TestSyntheaBundleFormat:
    """Verify that Synthea's transaction bundles with urn:uuid references work."""

    @pytest.fixture
    def synthea_store(self, tmp_path: Path) -> FHIRStore:
        """Build a minimal Synthea-style transaction bundle."""
        patient_uuid = "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"
        bundle = {
            "resourceType": "Bundle",
            "type": "transaction",
            "entry": [
                {
                    "fullUrl": f"urn:uuid:{patient_uuid}",
                    "resource": {
                        "resourceType": "Patient",
                        "id": patient_uuid,
                        "name": [{"use": "official", "family": "Santos", "given": ["Maria"]}],
                        "gender": "female",
                        "birthDate": "1985-07-20",
                    },
                    "request": {"method": "POST", "url": "Patient"},
                },
                {
                    "fullUrl": "urn:uuid:11111111-2222-3333-4444-555555555555",
                    "resource": {
                        "resourceType": "Condition",
                        "id": "11111111-2222-3333-4444-555555555555",
                        "code": {
                            "coding": [
                                {
                                    "system": "http://snomed.info/sct",
                                    "code": "44054006",
                                    "display": "Diabetes mellitus type 2",
                                }
                            ],
                            "text": "Diabetes mellitus type 2",
                        },
                        "subject": {"reference": f"urn:uuid:{patient_uuid}"},
                    },
                    "request": {"method": "POST", "url": "Condition"},
                },
            ],
        }
        path = tmp_path / "synthea_patient.json"
        path.write_text(json.dumps(bundle))

        s = FHIRStore()
        s.load_bundle(path)
        return s

    def test_patient_loaded(self, synthea_store: FHIRStore):
        patient_uuid = "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"
        patient = synthea_store.get_patient(patient_uuid)
        assert patient is not None
        assert patient["gender"] == "female"

    def test_condition_linked_via_urn_uuid(self, synthea_store: FHIRStore):
        patient_uuid = "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"
        conditions = synthea_store.get_conditions(patient_uuid)
        assert len(conditions) == 1
        assert "Diabetes" in conditions[0]["code"]["text"]

    def test_list_patients_name(self, synthea_store: FHIRStore):
        patients = synthea_store.list_patients()
        assert len(patients) == 1
        assert patients[0]["name"] == "Maria Santos"


# ------------------------------------------------------------------
# Multiple bundles
# ------------------------------------------------------------------


class TestMultipleBundles:
    """Verify that loading multiple bundles accumulates data correctly."""

    def test_two_bundles_two_patients(self, tmp_path: Path):
        for i, name in enumerate(["Ana", "Pedro"]):
            bundle = {
                "resourceType": "Bundle",
                "type": "collection",
                "entry": [
                    {
                        "resource": {
                            "resourceType": "Patient",
                            "id": f"patient-{i}",
                            "name": [{"family": "Test", "given": [name]}],
                        }
                    }
                ],
            }
            (tmp_path / f"p{i}.json").write_text(json.dumps(bundle))

        s = FHIRStore()
        s.load_directory(tmp_path)
        patients = s.list_patients()
        assert len(patients) == 2
        names = {p["name"] for p in patients}
        assert "Ana Test" in names
        assert "Pedro Test" in names


# ------------------------------------------------------------------
# Edge cases
# ------------------------------------------------------------------


class TestEdgeCases:
    """Edge cases and robustness checks."""

    def test_patient_without_name(self, tmp_path: Path):
        bundle = {
            "resourceType": "Bundle",
            "type": "collection",
            "entry": [{"resource": {"resourceType": "Patient", "id": "no-name"}}],
        }
        path = tmp_path / "no_name.json"
        path.write_text(json.dumps(bundle))
        s = FHIRStore()
        s.load_bundle(path)
        patients = s.list_patients()
        assert patients[0]["name"] == "Desconhecido"

    def test_resource_without_subject_is_ignored(self, tmp_path: Path):
        bundle = {
            "resourceType": "Bundle",
            "type": "collection",
            "entry": [
                {"resource": {"resourceType": "Patient", "id": "p1", "name": [{"given": ["X"]}]}},
                {
                    "resource": {
                        "resourceType": "Observation",
                        "id": "obs-no-subj",
                        "code": {"text": "test"},
                    }
                },
            ],
        }
        path = tmp_path / "no_subj.json"
        path.write_text(json.dumps(bundle))
        s = FHIRStore()
        s.load_bundle(path)
        # Observation without subject should not crash, just not be indexed
        assert s.get_observations("p1") == []

    def test_format_name_with_empty_name_array(self):
        assert FHIRStore._format_patient_name({"name": []}) == "Desconhecido"

    def test_format_name_with_given_only(self):
        assert FHIRStore._format_patient_name({"name": [{"given": ["Ana"]}]}) == "Ana"
