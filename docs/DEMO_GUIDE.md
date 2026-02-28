# Imaging Intelligence Agent -- Demo Guide

**Author:** Adam Jones
**Date:** February 2026
**Version:** 1.0.0

---

## 1. Demo Overview

This guide walks through five demonstration scenarios that showcase the Imaging Intelligence Agent's capabilities. All scenarios work in both Lite Mode (mock NIMs, no GPU) and Full Mode (live NIMs on DGX Spark).

**Demo duration:** 15-20 minutes for all five scenarios.

**Key talking points:**
- Multi-collection RAG across 10 imaging-specific knowledge domains
- On-device NVIDIA NIM inference for medical imaging (VISTA-3D, MAISI, VILA-M3)
- Clinical workflow automation with severity classification
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
      "description": "Acute intraparenchymal hemorrhage...",
      "severity": "critical",
      "recommendation": "Emergent neurosurgery consultation"
    }
  ],
  "measurements": {
    "volume_ml": 35.2,
    "midline_shift_mm": 6.8,
    "max_thickness_mm": 12.4
  },
  "classification": "P1",
  "severity": "critical",
  "inference_time_ms": 45.0,
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

**Goal:** Show clinical report export as PDF.

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
- Three output formats: markdown, JSON, PDF
- PDF generated via ReportLab with professional formatting
- Evidence citations include collection source and relevance scores
- Reports include a research-use-only disclaimer
- Suitable for clinical decision support documentation

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
