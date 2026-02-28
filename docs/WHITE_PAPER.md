# Medical Imaging AI on NVIDIA DGX Spark: An Integrated Intelligence Agent

**Author:** Adam Jones
**Date:** February 2026
**Version:** 1.0.0

---

## Abstract

We present the Imaging Intelligence Agent, an open-source medical imaging AI system that combines retrieval-augmented generation (RAG) with NVIDIA NIM inference microservices to deliver clinical decision support for radiology. The system integrates 10 domain-specific vector collections covering imaging literature, clinical trials, FDA-cleared devices, acquisition protocols, anatomical references, performance benchmarks, clinical guidelines, structured report templates, and public datasets -- all indexed with BGE-small-en-v1.5 embeddings in Milvus 2.4. Four NVIDIA NIM services (VISTA-3D, MAISI, VILA-M3, and Llama-3 8B) provide on-device 3D segmentation, synthetic CT generation, vision-language understanding, and clinical text synthesis. Four reference clinical workflows implement end-to-end analysis pipelines for CT head hemorrhage triage, CT chest lung nodule tracking, chest X-ray rapid findings, and MRI brain MS lesion quantification. The entire system deploys on a single NVIDIA DGX Spark ($3,999), making enterprise-grade medical imaging AI accessible to community hospitals, research institutions, and educational programs. Licensed under Apache 2.0, the platform is part of the HCLS AI Factory -- a precision medicine pipeline that processes patient DNA to drug candidates in under 5 hours.

---

## 1. Introduction

### 1.1 The Challenge

Medical imaging AI has matured from research prototypes to FDA-cleared products, with over 800 AI/ML-enabled medical devices authorized by the FDA as of 2025. Yet deployment remains concentrated in large academic medical centers with dedicated GPU infrastructure, specialized engineering teams, and significant capital investment. Community hospitals, rural health systems, and research institutions in low-resource settings lack the infrastructure and expertise to deploy, maintain, and integrate these models into clinical workflows.

### 1.2 Democratizing Medical Imaging AI

The NVIDIA DGX Spark addresses the infrastructure barrier with a $3,999 system featuring a GB10 GPU, 128 GB unified LPDDR5x memory, and 20 ARM cores -- sufficient to run multiple AI inference models simultaneously. The Imaging Intelligence Agent addresses the integration barrier by providing a complete, open-source platform that combines knowledge retrieval with inference in a single deployable package.

### 1.3 Design Goals

1. **Single-device deployment** -- Entire stack runs on one DGX Spark
2. **No cloud dependency** -- All inference happens on-device for data sovereignty
3. **Graceful degradation** -- Works without GPU in RAG-only mode
4. **Clinical relevance** -- Reference workflows address high-impact radiology use cases
5. **Open source** -- Apache 2.0 license enables institutional customization
6. **Cross-modal integration** -- Architecture supports triggers to genomics and drug discovery pipelines

---

## 2. Architecture

### 2.1 System Design

The Imaging Intelligence Agent is organized into four functional layers:

**Presentation Layer.** A Streamlit chat interface (port 8525) and FastAPI REST server (port 8524) provide the user-facing endpoints. The API exposes RAG queries, NIM proxy endpoints, workflow execution, and structured report generation (markdown, JSON, PDF).

**Intelligence Layer.** A multi-collection RAG engine searches 10 imaging-specific Milvus collections in parallel, applies weighted scoring, augments results with a domain knowledge graph (15 pathologies, 8 modalities, 15 anatomical structures), and synthesizes answers via Claude or Llama-3. The engine supports query expansion, comparative analysis (auto-detected "X vs Y" queries), and multi-turn conversation memory.

**Inference Layer.** Four NVIDIA NIM microservices provide GPU-accelerated inference for 3D segmentation (VISTA-3D), synthetic data generation (MAISI), vision-language understanding (VILA-M3), and text generation (Llama-3 8B). A unified NIM client layer provides health checking, retry logic, and automatic mock fallback. Four reference clinical workflows implement complete analysis pipelines.

**Data Layer.** Milvus 2.4 stores 384-dimensional BGE-small-en-v1.5 embeddings across 10 collections with IVF_FLAT indexing and cosine similarity. Ingest pipelines fetch from PubMed (NCBI E-utilities) and ClinicalTrials.gov (API v2), with seed data scripts for domain-specific knowledge.

### 2.2 RAG Pipeline

The RAG pipeline implements a five-stage process:

1. **Query expansion:** Domain-specific expansion maps broaden the search query with related medical terminology
2. **Parallel retrieval:** All collections are searched simultaneously using ThreadPoolExecutor, with per-collection top-K results
3. **Weighted merge:** Results are scored using configurable collection weights (literature: 0.18, findings: 0.15, trials: 0.12, guidelines: 0.10, etc.) and filtered by a minimum cosine similarity threshold of 0.4
4. **Knowledge augmentation:** Matched pathology, modality, and anatomy entries from the domain knowledge graph are injected as structured context
5. **LLM synthesis:** The question, retrieved evidence, and knowledge context are assembled into a prompt for Claude (Anthropic API) or Llama-3 (NIM), producing a grounded response with evidence citations

---

## 3. NVIDIA NIM Integration

### 3.1 NIM Architecture

Each NIM runs as an independent Docker container exposing an HTTP API. The agent communicates with NIMs through a client layer that provides:

- **Health checking:** Cached availability polling with a 30-second interval, using the standard `/v1/health/ready` endpoint
- **Retry logic:** Exponential-backoff retry via tenacity (3 attempts, 1-10 second wait)
- **Mock fallback:** When a NIM is unavailable and mock mode is enabled, the client returns clinically realistic synthetic responses
- **Unified interface:** All clients inherit from `BaseNIMClient`, providing consistent error handling and status reporting

### 3.2 VISTA-3D: 3D Medical Image Segmentation

NVIDIA VISTA-3D (Versatile Imaging Segmentation and Triage Accelerator) provides zero-shot 3D segmentation across 132 anatomical classes. In the Imaging Intelligence Agent, VISTA-3D supports:

- Automated organ segmentation for quantitative volume analysis
- Hemorrhage segmentation in the CT head hemorrhage workflow
- Lesion segmentation in the MRI brain MS lesion workflow
- Interactive mode allowing users to specify target anatomical classes

### 3.3 MAISI: Medical AI Synthetic Imaging

NVIDIA MAISI uses a latent diffusion model to generate high-resolution synthetic CT volumes (up to 512x512x512) with paired segmentation masks for up to 127 anatomical classes. Applications include:

- Training data augmentation for rare pathologies
- Algorithm validation without requiring patient data
- Educational demonstration of imaging findings

### 3.4 VILA-M3: Vision-Language Medical Model

VILA-M3 provides natural language understanding of medical images, combining visual feature extraction with medical domain knowledge. The model accepts an image and a natural language question, returning findings and a confidence score. Applications include:

- Automated preliminary radiology report drafting
- Interactive image-based question answering
- Quality assurance for existing reports

### 3.5 Memory Budget on DGX Spark

The 128 GB unified LPDDR5x memory of the DGX Spark enables simultaneous deployment of all four NIMs:

| NIM Service | Estimated Memory |
|---|---|
| Llama-3 8B Instruct | ~16 GB |
| VISTA-3D | ~8 GB |
| MAISI | ~12 GB |
| VILA-M3 | ~16 GB |
| Application stack | ~6 GB |
| **Total** | **~58 GB** |
| **Available headroom** | **~70 GB** |

This deployment model is not feasible on typical workstation GPUs (24-48 GB VRAM) but fits comfortably within the DGX Spark's unified memory architecture.

---

## 4. Knowledge System

### 4.1 Collection Design

The agent maintains 10 imaging-specific Milvus collections plus read-only access to the genomic evidence collection from the HCLS AI Factory's Stage 2 RAG pipeline. All collections use:

- **Embedding model:** BAAI/bge-small-en-v1.5 (384-dimensional, asymmetric query prefix)
- **Index type:** IVF_FLAT (nlist=1024, nprobe=16)
- **Similarity metric:** Cosine

| Collection | Content Domain | Seed Source |
|---|---|---|
| `imaging_literature` | Published research papers | PubMed (NCBI E-utilities) |
| `imaging_trials` | Clinical trials | ClinicalTrials.gov API v2 |
| `imaging_findings` | Imaging finding patterns | Curated seed data |
| `imaging_protocols` | Acquisition protocols | Curated seed data |
| `imaging_devices` | FDA-cleared AI/ML devices | Curated seed data |
| `imaging_anatomy` | Anatomical structures | Curated seed data |
| `imaging_benchmarks` | Model performance data | Curated seed data |
| `imaging_guidelines` | Clinical guidelines (ACR, RSNA) | Curated seed data |
| `imaging_report_templates` | Structured report templates | Curated seed data |
| `imaging_datasets` | Public datasets (TCIA, PhysioNet) | Curated seed data |

### 4.2 Ingest Pipelines

Two automated ingest pipelines maintain currency:

**PubMed ingest** fetches medical imaging AI abstracts via NCBI E-utilities (esearch + efetch), extracts modality, body region, AI task, and disease fields, embeds with BGE-small, and stores in `imaging_literature`. Supports an optional NCBI API key for increased rate limits. Configurable maximum results (default 5,000).

**ClinicalTrials.gov ingest** fetches AI-in-imaging trials via the v2 API, extracts phase, status, sponsor, modality, enrollment, and outcome data, embeds, and stores in `imaging_trials`.

Both pipelines support scheduled execution via APScheduler (default weekly, configurable).

### 4.3 Domain Knowledge Graph

A structured knowledge graph provides RAG augmentation beyond vector retrieval:

- **15 pathology entries:** ICD-10 codes, imaging characteristics (CT and MRI), severity criteria, key measurements, clinical guidelines, and associated AI models
- **8 modality entries:** Physics principles, typical protocols, clinical indications, strengths and limitations
- **15 anatomy entries:** SNOMED codes, Foundational Model of Anatomy (FMA) IDs, imaging characteristics, common pathologies, and VISTA-3D segmentation label IDs

---

## 5. Reference Clinical Workflows

### 5.1 CT Head Hemorrhage Triage

An emergency radiology workflow for automated intracranial hemorrhage detection and triage. The pipeline applies a 3D U-Net (MONAI) for binary hemorrhage segmentation on CT head volumes preprocessed to RAS orientation, 1mm isotropic spacing, and blood window (0-80 HU). Postprocessing computes hemorrhage volume (mL), midline shift (mm), and maximum thickness (mm). Urgency classification follows Brain Trauma Foundation thresholds: volume > 30 mL or shift > 5 mm or thickness > 10 mm triggers Critical (P1) priority; volume > 5 mL triggers Urgent (P2). Target end-to-end latency is under 90 seconds.

### 5.2 CT Chest Lung Nodule Tracking

A lung cancer screening workflow implementing the ACR Lung-RADS v2022 classification. Detection uses a RetinaNet (MONAI) to identify candidate nodules, followed by SegResNet per-nodule segmentation for volumetric analysis. Longitudinal tracking retrieves prior CT chest studies, applies SyN diffeomorphic registration, and computes volume doubling time (VDT). Rule-based Lung-RADS assignment considers nodule morphology (solid, ground-glass, part-solid), size, and growth rate. A cross-modal trigger mechanism routes Lung-RADS 4B+ findings to the Parabricks genomics pipeline for tumor profiling. Target latency is under 5 minutes including prior retrieval and registration.

### 5.3 CXR Rapid Findings

A triage workflow for chest X-ray multi-label classification using DenseNet-121. The model detects consolidation, effusion, pneumothorax, cardiomegaly, atelectasis, edema, and nodules with GradCAM heatmap localization for visual explanation. Target latency is under 30 seconds, making it suitable for emergency department workflows where rapid triage is critical.

### 5.4 MRI Brain MS Lesion Tracking

A multiple sclerosis monitoring workflow using 3D U-Net segmentation on FLAIR sequences. The pipeline includes SyN diffeomorphic registration for longitudinal comparison, lesion matching to classify new, enlarged, stable, and resolved lesions, and quantitative metrics including total lesion volume, lesion count, and lesion change rate. Target latency is under 5 minutes.

### 5.5 Workflow Implementation Pattern

All workflows inherit from `BaseImagingWorkflow` and implement a three-stage pipeline:

1. **Preprocess:** Load DICOM/NIfTI, reorient, resample, window/normalize
2. **Infer:** Run model inference (real via MONAI/NIM or mock with synthetic data)
3. **Postprocess:** Extract measurements, apply classification rules, determine severity

Mock mode returns clinically realistic synthetic results for demonstration and testing without requiring GPU or medical image data.

---

## 6. DGX Spark Deployment

### 6.1 Hardware Specification

| Component | Specification |
|---|---|
| GPU | NVIDIA GB10 (Blackwell architecture) |
| Memory | 128 GB unified LPDDR5x |
| CPU | 20 ARM cores (Grace architecture) |
| Interconnect | NVLink-C2C (CPU-GPU) |
| Price | $3,999 |

### 6.2 Deployment Models

**Full deployment** runs all services including four NIM containers, consuming approximately 58 GB of the 128 GB unified memory. This configuration provides complete on-device inference without cloud dependency.

**Lite deployment** runs the RAG knowledge system (Milvus + application stack) without NIM containers. NIM-dependent features operate in mock mode. This configuration requires no GPU and runs on any machine with Docker, making it accessible for development, testing, and demonstration.

### 6.3 Data Sovereignty

All inference happens on-device. No patient data, imaging studies, or clinical queries leave the DGX Spark. This architecture supports deployment in regulated environments (HIPAA, GDPR) where cloud-based inference may face compliance barriers.

---

## 7. Results and Performance

### 7.1 RAG Performance

| Metric | Value |
|---|---|
| Vector search (11 collections, top-5 each) | 12-16 ms (warm cache) |
| Query expansion | < 1 ms |
| Knowledge graph augmentation | < 1 ms |
| Full RAG query (search + Claude synthesis) | ~20-30 sec |
| Comparative RAG query (dual retrieval) | ~25-35 sec |
| Cosine similarity scores (typical) | 0.70-0.90 |

### 7.2 Workflow Target Latencies

| Workflow | Target | Mode |
|---|---|---|
| CT Head Hemorrhage Triage | < 90 seconds | Live inference |
| CT Chest Lung Nodule Tracking | < 5 minutes | Live inference |
| CXR Rapid Findings | < 30 seconds | Live inference |
| MRI Brain MS Lesion Tracking | < 5 minutes | Live inference |
| Any workflow (mock mode) | < 100 ms | Mock |

### 7.3 NIM Startup Times

| NIM Service | Model Load Time |
|---|---|
| Llama-3 8B Instruct | 60-90 seconds |
| VISTA-3D | 60-90 seconds |
| MAISI | 90-120 seconds |
| VILA-M3 | 60-90 seconds |

---

## 8. Future Directions

### 8.1 Phase 2 Roadmap

**DICOM server integration.** Orthanc DICOM server for STOW-RS/WADO-RS, enabling direct integration with PACS and modality worklists. Event-driven webhook triggers on study completion to initiate automatic workflow execution.

**Cross-modal pipeline triggers.** Lung-RADS 4B+ findings automatically trigger the Parabricks genomics pipeline for tumor profiling. Quantitative imaging endpoints (RECIST measurements, tumor volume changes) feed into the drug discovery pipeline for treatment-response tracking.

**FHIR R4 output.** DiagnosticReport and Observation resources for EHR integration, enabling automated result reporting and clinical decision support alerts.

**DICOM SR output.** Structured reports via highdicom TID 1500 measurement reports, stored directly in the DICOM archive alongside source images.

**LangGraph multi-step agent.** An advanced reasoning agent that chains triage, longitudinal analysis, population comparison, and report generation in a multi-step workflow with human-in-the-loop checkpoints.

### 8.2 Research Directions

**Federated learning.** Privacy-preserving model training across multiple institutions using the DGX Spark as a local training node, enabling model improvement without centralizing patient data.

**Population analytics.** Cohort-level imaging trends, disease prevalence monitoring, and outcomes tracking across institutional imaging archives.

**Multimodal integration.** Combining imaging findings with genomic variants, pathology results, and clinical notes for comprehensive patient assessment.

---

## 9. Conclusion

The Imaging Intelligence Agent demonstrates that enterprise-grade medical imaging AI can run on a single $3,999 device. By combining RAG-grounded knowledge retrieval with NVIDIA NIM on-device inference, the system provides clinical decision support that is evidence-based, locally deployed, and open source. The platform's graceful degradation ensures accessibility -- organizations without GPU infrastructure can still leverage the RAG knowledge system in Lite Mode, adopting NIM inference as hardware becomes available.

As part of the HCLS AI Factory platform, the Imaging Intelligence Agent connects to genomics (Parabricks) and drug discovery (BioNeMo) pipelines, enabling a precision medicine workflow from patient DNA to drug candidates -- all on a single NVIDIA DGX Spark.

---

## References

1. NVIDIA DGX Spark Technical Specifications, NVIDIA Corporation, 2025
2. ACR Lung-RADS v2022, American College of Radiology, 2022
3. Brain Trauma Foundation Guidelines, 4th Edition, 2016
4. Milvus: A Purpose-Built Vector Data Management System, Wang et al., SIGMOD 2021
5. BGE-small-en-v1.5, BAAI, 2023
6. MONAI: Medical Open Network for Artificial Intelligence, Project MONAI Consortium
7. VISTA-3D: Versatile Imaging Segmentation and Triage Accelerator, NVIDIA
8. MAISI: Medical AI Synthetic Imaging, NVIDIA
9. VILA-M3: Vision-Language Model for Medical Imaging, NVIDIA

---

*Apache 2.0 License. HCLS AI Factory -- Imaging Intelligence Agent.*
