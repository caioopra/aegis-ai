#!/usr/bin/env bash
# generate_synthea.sh — Download Synthea and generate synthetic FHIR patient records.
#
# Synthea is an open-source patient generator that creates realistic (but fictional)
# patient records in FHIR format. FHIR (Fast Healthcare Interoperability Resources)
# is the international standard for exchanging healthcare data electronically.
#
# Each generated patient comes as a "Bundle" — a single JSON file containing all
# their clinical data: demographics, conditions, medications, vital signs, etc.
#
# Usage:
#   ./scripts/generate_synthea.sh [NUM_PATIENTS]
#
# Examples:
#   ./scripts/generate_synthea.sh        # Generate 5 patients (default)
#   ./scripts/generate_synthea.sh 20     # Generate 20 patients

set -euo pipefail

NUM_PATIENTS="${1:-5}"
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SYNTHEA_DIR="${PROJECT_ROOT}/.synthea"
OUTPUT_DIR="${PROJECT_ROOT}/data/synthea"

echo "=== AegisNode: Synthea Patient Generator ==="
echo "Patients to generate: ${NUM_PATIENTS}"
echo ""

# ── Step 1: Download Synthea if not present ──────────────────────────────
if [ ! -f "${SYNTHEA_DIR}/synthea-with-dependencies.jar" ]; then
    echo "Downloading Synthea..."
    mkdir -p "${SYNTHEA_DIR}"
    curl -L -o "${SYNTHEA_DIR}/synthea-with-dependencies.jar" \
        "https://github.com/synthetichealth/synthea/releases/download/master-branch-latest/synthea-with-dependencies.jar"
    echo "Download complete."
else
    echo "Synthea already downloaded."
fi

# ── Step 2: Generate patients ────────────────────────────────────────────
# Synthea flags:
#   -p N           → generate N patients
#   --exporter.fhir.export true  → output FHIR R4 JSON bundles
#   --exporter.hospital.fhir.export false → skip hospital bundles
#   --exporter.practitioner.fhir.export false → skip practitioner bundles
#   --exporter.baseDirectory → where to write output
#
# Note: Synthea doesn't have a native Brazilian locale, so records will have
# English-structured data. The MCP server layer will handle any adaptation.

mkdir -p "${OUTPUT_DIR}"

echo ""
echo "Generating ${NUM_PATIENTS} patients..."
java -jar "${SYNTHEA_DIR}/synthea-with-dependencies.jar" \
    -p "${NUM_PATIENTS}" \
    --exporter.fhir.export=true \
    --exporter.hospital.fhir.export=false \
    --exporter.practitioner.fhir.export=false \
    --exporter.baseDirectory="${OUTPUT_DIR}" \
    Massachusetts

# Synthea outputs to a "fhir/" subdirectory — move files up and clean
if [ -d "${OUTPUT_DIR}/fhir" ]; then
    mv "${OUTPUT_DIR}"/fhir/*.json "${OUTPUT_DIR}/" 2>/dev/null || true
    rm -rf "${OUTPUT_DIR}/fhir"
fi

# Remove hospital/practitioner info files if generated
rm -f "${OUTPUT_DIR}"/hospitalInformation*.json
rm -f "${OUTPUT_DIR}"/practitionerInformation*.json

GENERATED=$(ls "${OUTPUT_DIR}"/*.json 2>/dev/null | wc -l)
echo ""
echo "Done! Generated ${GENERATED} patient bundles in: ${OUTPUT_DIR}/"
echo "Each file is a FHIR Bundle containing a patient's full clinical history."
