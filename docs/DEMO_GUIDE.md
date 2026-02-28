# Imaging Intelligence Agent -- Demo Guide

**Author:** Adam Jones
**Date:** February 2026
**Version:** 1.0.0

---

## 1. Demo Overview

This guide walks through eight demonstration scenarios that showcase the Imaging Intelligence Agent's capabilities. All scenarios work in both Lite Mode (mock NIMs, no GPU) and Full Mode (live NIMs on DGX Spark).

**Demo duration:** 25-30 minutes for all eight scenarios.

**Key talking points:**
- Multi-collection RAG across 10 imaging-specific knowledge domains
- On-device NVIDIA NIM inference for medical imaging (VISTA-3D, MAISI, VILA-M3)
- Clinical workflow automation with severity classification
- FHIR R4 DiagnosticReport export with SNOMED CT and LOINC coding
- Orthanc DICOM auto-ingestion with intelligent workflow routing
- Cross-modal genomics integration (imaging → genomic variant correlation)
- NVIDIA FLARE federated learning configuration
- Runs entirely on a single NVIDIA DGX Spark ($3,999)
- Open source, Apache 2.0 licensed

---

## 2. Setup

### 2.1 Start Services

**Lite Mode (recommended for first demo):**

```bash
cd ai_agent_adds/imaging_intelligence_agent/agent
cp .env.example .env
# Edit .env: set ANTHROPIC_API_KEY

docker compose -f docker-compose.lite.yml up -d
# Wait for setup to complete:
docker compose -f docker-compose.lite.yml logs -f imaging-setup
```

**Full Mode (with NIMs on GPU):**

```bash
cd ai_agent_adds/imaging_intelligence_agent/agent
cp .env.example .env
# Edit .env: set ANTHROPIC_API_KEY and NGC_API_KEY

docker compose up -d
# Wait for NIMs to initialize (60-120 seconds):
docker compose logs -f nim-llm
```

### 2.2 Verify Services

```bash
# Check health
curl http://localhost:8524/health | python -m json.tool

# Verify collection data
curl http://localhost:8524/collections | python -m json.tool
```

**Expected health output:**
```json
{
  "status": "healthy",
  "collections": {
    "imaging_literature": 2678,
    "imaging_trials": 12,
    "imaging_findings": 25,
    "imaging_protocols": 15,
    "imaging_devices": 15,
    "imaging_anatomy": 20,
    "imaging_benchmarks": 15,
    "imaging_guidelines": 12,
    "imaging_report_templates": 10,
    "imaging_datasets": 12,
    "genomic_evidence": 3561170
  },
  "total_vectors": 3563984,
  "nim_services": {
    "vista3d": "mock",
    "maisi": "mock",
    "vila_m3": "cloud",
    "llm": "cloud"
  }
}
```

### 2.3 Open the UI

Navigate to [http://localhost:8525](http://localhost:8525) in your browser.

---

## 3. Demo Scenarios

### Scenario 1: RAG Knowledge Query

**Goal:** Show multi-collection retrieval with grounded, cited answers.

**Query:**
```
What is ACR Lung-RADS classification for lung nodule management?
```

**Expected behavior:**
1. The agent searches across all 10+ collections simultaneously
2. High-relevance hits from `imaging_guidelines` (Lung-RADS v2022), `imaging_findings` (nodule patterns), and `imaging_literature`
3. Claude synthesizes a grounded answer explaining Lung-RADS categories (0-4X, S)
4. Evidence panel shows collection badges with cosine similarity scores
5. Follow-up question suggestions appear (e.g., "How does volume doubling time help differentiate benign from malignant nodules?")

**Talking points:**
- 10 specialized collections ensure comprehensive coverage
- Weighted scoring prioritizes guidelines and findings for clinical queries
- All answers are grounded in retrieved evidence with citations
- Knowledge graph augments with structured pathology data (Lung-RADS categories, severity criteria)

---

### Scenario 2: Comparative Analysis

**Goal:** Show auto-detected comparative mode with dual retrieval.

**Query:**
```
Compare CT vs MRI for detecting brain metastases
```

**Expected behavior:**
1. The agent auto-detects the comparative pattern ("CT vs MRI")
2. Runs dual retrieval: one search for CT-related evidence, one for MRI-related evidence
3. Claude produces a structured comparison with a markdown table
4. Evidence panel groups results by entity (CT evidence and MRI evidence)
5. Answer covers sensitivity, specificity, advantages, and limitations of each modality

**Talking points:**
- Comparative queries are automatically detected via keyword parsing ("vs", "versus", "compare")
- Dual retrieval runs two parallel searches with entity-specific filtering
- Structured output includes comparison tables, advantages, limitations, and clinical context
- Same pattern used in the CAR-T Intelligence Agent for target antigen comparisons

---

### Scenario 3: Workflow Demo (CT Head Hemorrhage Triage)

**Goal:** Show clinical workflow execution with severity classification.

**Using the API:**
```bash
curl -X POST http://localhost:8524/workflow/ct_head_hemorrhage/run \
  -H "Content-Type: application/json" \
  -d '{"input_path": "", "mock_mode": true}'
```

**Using the UI:**
1. In the Streamlit sidebar, find the "Workflow Runner" section
2. Select "ct_head_hemorrhage" from the dropdown
3. Click "Run Workflow"

**Expected output:**
```json
{
  "workflow_name": "ct_head_hemorrhage",
  "status": "completed",
  "findings": [
    {
      "category": "hemorrhage",
      "description": "Intraparenchymal hemorrhage in right basal ganglia, volume 12.5 mL, midline shift 3.2 mm, max thickness 8.1 mm",
      "severity": "urgent"
    }
  ],
  "measurements": {
    "volume_ml": 12.5,
    "midline_shift_mm": 3.2,
    "max_thickness_mm": 8.1,
    "hounsfield_mean": 62.0,
    "hounsfield_max": 78.0,
    "surrounding_edema_ml": 4.3
  },
  "classification": "urgent_hemorrhage",
  "severity": "urgent",
  "inference_time_ms": 0.1,
  "is_mock": true
}
```

**Talking points:**
- Follows the Brain Trauma Foundation severity thresholds
- Volume > 30 mL OR shift > 5 mm = Critical (P1)
- Pipeline: preprocess -> infer (3D U-Net) -> postprocess -> severity classification
- Mock mode returns clinically realistic results for demonstration
- In Full Mode, uses real MONAI inference on actual DICOM data

---

### Scenario 4: NIM Demo (VISTA-3D Segmentation)

**Goal:** Show NVIDIA NIM service integration (mock or live).

**Check NIM status first:**
```bash
curl http://localhost:8524/nim/status | python -m json.tool
```

**Run VISTA-3D segmentation:**
```bash
curl -X POST http://localhost:8524/nim/vista3d/segment \
  -H "Content-Type: application/json" \
  -d '{"input_path": "", "target_classes": ["liver", "spleen", "left_kidney", "right_kidney"]}'
```

**Expected output (mock mode):**
```json
{
  "classes_detected": ["liver", "spleen", "left_kidney", "right_kidney"],
  "volumes": {
    "liver": 1450.2,
    "spleen": 180.5,
    "left_kidney": 160.3,
    "right_kidney": 155.8
  },
  "num_classes": 4,
  "inference_time_ms": 45.2,
  "model": "vista3d",
  "is_mock": true
}
```

**Talking points:**
- VISTA-3D supports 132 anatomical classes for zero-shot segmentation
- Mock mode returns realistic volumes for any requested anatomy
- In Full Mode, processes real NIfTI volumes via the VISTA-3D NIM container
- All NIM clients use cached health checks and automatic mock fallback
- On DGX Spark, all four NIMs run simultaneously in 128 GB unified memory

**If NIMs are live (Full Mode), also demonstrate:**
```bash
# MAISI synthetic CT generation
curl -X POST http://localhost:8524/nim/maisi/generate \
  -H "Content-Type: application/json" \
  -d '{"body_region": "chest", "resolution": "512x512x512"}'

# VILA-M3 image analysis
curl -X POST http://localhost:8524/nim/vilam3/analyze \
  -H "Content-Type: application/json" \
  -d '{"question": "Describe the findings in this chest CT"}'
```

---

### Scenario 5: Report Generation

**Goal:** Show clinical report export as PDF and FHIR R4.

**Using the API:**
```bash
# Generate markdown report
curl -X POST http://localhost:8524/reports/generate \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What are the imaging features of multiple sclerosis on MRI?",
    "format": "markdown",
    "top_k": 5,
    "include_evidence": true
  }'

# Generate PDF report (downloads file)
curl -X POST http://localhost:8524/reports/generate \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What are the imaging features of multiple sclerosis on MRI?",
    "format": "pdf",
    "top_k": 5,
    "report_title": "MS Imaging Features Report"
  }' --output ms_report.pdf

# Generate FHIR R4 DiagnosticReport (via Python)
python3 -c "
from src.export import export_fhir
# ... (after running a query/workflow)
fhir_json = export_fhir(response, patient_id='DEMO-001')
print(fhir_json[:500])
"
```

**Using the UI:**
1. Submit any clinical question in the chat
2. After the answer appears, click "Export as PDF" in the results panel
3. A PDF file downloads with the clinical question, RAG-synthesized analysis, and evidence citations

**Expected PDF contents:**
- Report title and timestamp
- Clinical question
- RAG-synthesized analysis with clinical reasoning
- Evidence citations with collection labels and relevance scores
- Disclaimer noting research use only

**Talking points:**
- Four output formats: markdown, JSON, PDF, FHIR R4 DiagnosticReport
- PDF generated via ReportLab with professional formatting
- Evidence citations include collection source and relevance scores
- Reports include a research-use-only disclaimer
- Suitable for clinical decision support documentation

---

### Scenario 6: FHIR R4 DiagnosticReport Export

**Goal:** Show FHIR R4 interoperability with standardized clinical coding.

**Using the API:**
```bash
# Run a workflow to get clinical findings
curl -s -X POST http://localhost:8524/workflow/ct_head_hemorrhage/run \
  -H "Content-Type: application/json" \
  -d '{"input_path": ""}' | python3 -m json.tool
```

**Expected FHIR Bundle structure:**
```
Bundle (type: collection)
├── Patient (id: DEMO-001)
├── ImagingStudy (modality: CT)
├── Observation (SNOMED: 50960005 Hemorrhage, interpretation: High)
│   └── 6 measurement components (volume_ml, midline_shift_mm, etc.)
└── DiagnosticReport
    ├── LOINC Category: LP29684-5 (Radiology)
    ├── LOINC Code: 18748-4 (Diagnostic imaging study)
    ├── SNOMED conclusionCode: 50960005 (Hemorrhage)
    └── Performer: AI-ImagingAgent
```

**Talking points:**
- FHIR R4 is the global healthcare interoperability standard
- SNOMED CT codes for 10+ imaging finding categories
- LOINC coding for radiology category and diagnostic imaging study
- DICOM modality codes automatically detected from query context
- Bundle can be submitted directly to FHIR-compliant EHR systems
- Observation resources include measurement components with UCUM units

---

### Scenario 7: DICOM Webhook Auto-Routing

**Goal:** Show automatic clinical workflow triggering from DICOM study events.

**Check routing table:**
```bash
curl -s http://localhost:8524/events/status | python3 -m json.tool
```

**Simulate incoming DICOM studies:**
```bash
# CT Head study -> auto-triggers hemorrhage workflow
curl -s -X POST http://localhost:8524/events/dicom-webhook \
  -H "Content-Type: application/json" \
  -d '{"event_type":"study.complete","study_uid":"1.2.840.1001","patient_id":"P-ICH-001","modality":"CT","body_region":"head"}'

# Chest X-ray -> auto-triggers CXR rapid findings
curl -s -X POST http://localhost:8524/events/dicom-webhook \
  -H "Content-Type: application/json" \
  -d '{"event_type":"study.complete","study_uid":"1.2.840.1002","patient_id":"P-CXR-002","modality":"CR","body_region":"chest"}'

# Check event history
curl -s http://localhost:8524/events/history?limit=5 | python3 -m json.tool
```

**Expected routing:**

| Modality + Body Region | Workflow Triggered |
|---|---|
| CT + head | ct_head_hemorrhage |
| CT + chest | ct_chest_lung_nodule |
| CR/DX + chest | cxr_rapid_findings |
| MR + brain | mri_brain_ms_lesion |

**Talking points:**
- Orthanc DICOM server receives studies via C-STORE (port 4242)
- Webhook fires on study.complete event
- Intelligent routing based on modality + body region
- Supports 8 routing rules covering all 4 workflows
- Event history tracks all processed studies with results
- In production, connects to hospital PACS via standard DICOM protocol

---

### Scenario 8: Cross-Modal Genomics Integration

**Goal:** Show automatic genomic variant correlation when high-risk findings detected.

**Run lung nodule workflow (triggers cross-modal):**
```bash
curl -s -X POST http://localhost:8524/workflow/ct_chest_lung_nodule/run \
  -H "Content-Type: application/json" \
  -d '{"input_path": ""}' | python3 -c "
import json, sys
r = json.load(sys.stdin)
print(f'Classification: {r[\"classification\"]}')
print(f'Severity: {r[\"severity\"]}')
cm = r.get('cross_modal', {})
if cm:
    print(f'Cross-modal triggered: {cm[\"trigger_reason\"]}')
    print(f'Genomic queries: {cm[\"query_count\"]}')
    print(f'Genomic hits: {cm[\"genomic_hit_count\"]}')
    for ctx in cm.get('genomic_context', [])[:3]:
        print(f'  - {ctx[:100]}...')
"
```

**Expected behavior:**
1. Lung nodule workflow classifies nodules (Lung-RADS 4A)
2. Lung-RADS 4A triggers cross-modal enrichment
3. Agent queries genomic_evidence collection for EGFR, ALK, ROS1, KRAS variants
4. 12 genomic hits returned, including real EGFR variant at chr7:55181370
5. Response includes both imaging findings and genomic context

**Talking points:**
- Demonstrates the HCLS AI Factory's cross-pipeline intelligence
- Imaging findings automatically correlate with genomic variants from Stage 2 RAG pipeline
- 3.5 million genomic evidence vectors from real patient data (VCF → ClinVar → AlphaMissense)
- Threshold-based triggering: only high-risk findings (Lung-RADS 4A+) activate genomic queries
- Enables precision medicine workflow: imaging → genomics → targeted therapy

---

## 4. Troubleshooting

### Services not responding

```bash
# Check all containers are running
docker compose ps

# Check for errors in logs
docker compose logs imaging-api
docker compose logs imaging-streamlit
```

### No data in collections

```bash
# Re-run the setup container
docker compose run --rm imaging-setup
```

### Streamlit UI shows connection error

Verify the API is healthy:
```bash
curl http://localhost:8524/health
```

If the API returns a `503`, Milvus may still be initializing. Wait 30-60 seconds and retry.

### NIM services show "unavailable" (Full Mode)

NIM containers require 60-120 seconds to load models after starting. Check logs:
```bash
docker compose logs --tail 20 nim-llm
```

Look for a "ready" or "server started" message. If NIMs fail to start, verify `NGC_API_KEY` is set and the GPU is available (`nvidia-smi`).

### Queries return empty results

Verify collections have data:
```bash
curl http://localhost:8524/collections | python -m json.tool
```

If all counts are 0, the setup container did not complete. Check:
```bash
docker compose logs imaging-setup
```

---

*For NIM-specific troubleshooting, see `NIM_INTEGRATION_GUIDE.md`. For the full architecture, see `ARCHITECTURE_GUIDE.md`.*
