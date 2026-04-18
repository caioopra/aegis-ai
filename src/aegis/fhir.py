"""FHIR Bundle loader and resource lookup helpers."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from aegis.config import settings

# Type alias for a FHIR resource dict
Resource = dict[str, Any]


class FHIRStore:
    """In-memory store for FHIR resources loaded from Bundle JSON files.

    Handles both hand-crafted sample bundles (type=collection, references
    like ``Patient/id``) and real Synthea bundles (type=transaction,
    references like ``urn:uuid:id``).
    """

    def __init__(self) -> None:
        self._patients: dict[str, Resource] = {}
        self._patient_index: dict[str, dict[str, list[Resource]]] = {}
        # Maps any reference string (fullUrl, Patient/id) → patient_id
        self._ref_to_patient_id: dict[str, str] = {}

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def load_bundle(self, path: Path) -> None:
        """Load a single FHIR Bundle JSON file into the store."""
        with open(path) as f:
            bundle = json.load(f)

        if bundle.get("resourceType") != "Bundle":
            raise ValueError(f"Not a FHIR Bundle: {path}")

        entries = bundle.get("entry", [])

        # First pass: register patients and build reference map
        for entry in entries:
            resource = entry.get("resource", {})
            if resource.get("resourceType") == "Patient":
                patient_id = resource["id"]
                self._patients[patient_id] = resource
                self._patient_index.setdefault(patient_id, {})

                full_url = entry.get("fullUrl", "")
                if full_url:
                    self._ref_to_patient_id[full_url] = patient_id
                self._ref_to_patient_id[f"Patient/{patient_id}"] = patient_id

        # Second pass: index non-Patient resources by patient
        for entry in entries:
            resource = entry.get("resource", {})
            res_type = resource.get("resourceType", "")
            if res_type == "Patient":
                continue

            patient_id = self._resolve_patient_id(resource)
            if patient_id and patient_id in self._patient_index:
                self._patient_index[patient_id].setdefault(res_type, []).append(resource)

    def load_directory(self, directory: Path | None = None) -> None:
        """Load all ``*.json`` Bundle files from a directory."""
        directory = directory or settings.synthea_data_dir
        for path in sorted(directory.glob("*.json")):
            # Skip metadata files
            if path.parent.name == "metadata":
                continue
            self.load_bundle(path)

    # ------------------------------------------------------------------
    # Lookups
    # ------------------------------------------------------------------

    def list_patients(self) -> list[dict[str, str]]:
        """Return ``[{"id": ..., "name": ...}, ...]`` for all loaded patients."""
        return [
            {"id": pid, "name": self._format_patient_name(patient)}
            for pid, patient in self._patients.items()
        ]

    def get_patient(self, patient_id: str) -> Resource | None:
        """Return the Patient resource for the given ID, or ``None``."""
        return self._patients.get(patient_id)

    def get_resources(self, patient_id: str, resource_type: str) -> list[Resource]:
        """Return all resources of *resource_type* for a patient."""
        return self._patient_index.get(patient_id, {}).get(resource_type, [])

    def get_conditions(self, patient_id: str) -> list[Resource]:
        """Return Condition resources for a patient."""
        return self.get_resources(patient_id, "Condition")

    def get_medications(self, patient_id: str) -> list[Resource]:
        """Return MedicationRequest resources for a patient."""
        return self.get_resources(patient_id, "MedicationRequest")

    def get_observations(self, patient_id: str) -> list[Resource]:
        """Return Observation resources for a patient."""
        return self.get_resources(patient_id, "Observation")

    def get_procedures(self, patient_id: str) -> list[Resource]:
        """Return Procedure resources for a patient."""
        return self.get_resources(patient_id, "Procedure")

    def get_diagnostic_reports(self, patient_id: str) -> list[Resource]:
        """Return DiagnosticReport resources for a patient."""
        return self.get_resources(patient_id, "DiagnosticReport")

    def get_encounters(self, patient_id: str) -> list[Resource]:
        """Return Encounter resources for a patient."""
        return self.get_resources(patient_id, "Encounter")

    def get_immunizations(self, patient_id: str) -> list[Resource]:
        """Return Immunization resources for a patient."""
        return self.get_resources(patient_id, "Immunization")

    def get_allergy_intolerances(self, patient_id: str) -> list[Resource]:
        """Return AllergyIntolerance resources for a patient."""
        return self.get_resources(patient_id, "AllergyIntolerance")

    def get_patient_by_cpf(self, cpf: str) -> Resource | None:
        """Lookup a patient by CPF (Brazilian tax ID).

        Normalizes the CPF by stripping dots and dashes before comparison.
        Returns None if no patient has a matching identifier with the
        Brazilian CPF OID (urn:oid:2.16.840.1.113883.13.237).
        """
        if not cpf:
            return None
        cpf_digits = re.sub(r"\D", "", cpf)
        if not cpf_digits:
            return None
        for patient in self._patients.values():
            for identifier in patient.get("identifier", []):
                if identifier.get("system") == "urn:oid:2.16.840.1.113883.13.237":
                    stored_digits = re.sub(r"\D", "", identifier.get("value", ""))
                    if stored_digits == cpf_digits:
                        return patient
        return None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _resolve_patient_id(self, resource: Resource) -> str | None:
        """Extract the patient ID from a resource's ``subject`` or ``patient`` reference."""
        ref = resource.get("subject", {}).get("reference", "")
        if not ref:
            # Immunization uses "patient" instead of "subject"
            ref = resource.get("patient", {}).get("reference", "")
        if not ref:
            return None

        if ref in self._ref_to_patient_id:
            return self._ref_to_patient_id[ref]

        # Fallback: parse "Patient/<id>" directly
        if ref.startswith("Patient/"):
            return ref.split("/", 1)[1]

        return None

    @staticmethod
    def _format_patient_name(patient: Resource) -> str:
        """Format the first name entry as ``'Given Family'``."""
        names = patient.get("name", [])
        if not names:
            return "Desconhecido"
        name = names[0]
        given = " ".join(name.get("given", []))
        family = name.get("family", "")
        return f"{given} {family}".strip() or "Desconhecido"


_shared_store: FHIRStore | None = None


def get_store() -> FHIRStore:
    """Return the shared FHIRStore singleton, loading data on first call."""
    global _shared_store
    if _shared_store is None:
        _shared_store = FHIRStore()
        _shared_store.load_directory()
    return _shared_store
