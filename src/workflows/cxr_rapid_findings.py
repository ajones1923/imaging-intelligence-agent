"""CXR Rapid Findings Multi-Label Classification Workflow.

Reference workflow #3: Emergency chest X-ray triage.
Uses DenseNet-121 (MONAI) for multi-label classification of critical findings
with optional VILA-M3 vision-language model for natural language report generation.
Target: sub-30-second triage for emergency department workflow.
"""

from typing import Any, Dict, List

from loguru import logger

from src.models import FindingCategory, FindingSeverity, WorkflowResult, WorkflowStatus
from src.workflows.base import BaseImagingWorkflow


# Per-class confidence thresholds (tuned for high specificity in triage)
CXR_CLASS_THRESHOLDS = {
    "pneumothorax": 0.50,
    "consolidation": 0.60,
    "pleural_effusion": 0.55,
    "cardiomegaly": 0.60,
    "fracture": 0.55,
}

# Map CXR classes to finding categories
CXR_CLASS_TO_CATEGORY = {
    "pneumothorax": FindingCategory.PNEUMOTHORAX,
    "consolidation": FindingCategory.CONSOLIDATION,
    "pleural_effusion": FindingCategory.EFFUSION,
    "cardiomegaly": FindingCategory.NORMAL,  # No dedicated cardiomegaly enum; uses NORMAL
    "fracture": FindingCategory.FRACTURE,
}

# Per-class severity when positive (base severity before confidence adjustment)
CXR_CLASS_SEVERITY = {
    "pneumothorax": FindingSeverity.CRITICAL,
    "consolidation": FindingSeverity.URGENT,
    "pleural_effusion": FindingSeverity.SIGNIFICANT,
    "cardiomegaly": FindingSeverity.ROUTINE,
    "fracture": FindingSeverity.URGENT,
}

# Per-class clinical descriptions
CXR_CLASS_DESCRIPTIONS = {
    "pneumothorax": (
        "Pneumothorax detected. Evaluate for tension pneumothorax. "
        "Consider chest tube placement if large or symptomatic."
    ),
    "consolidation": (
        "Pulmonary consolidation identified, suggestive of pneumonia or "
        "atelectasis. Clinical correlation with infection markers recommended."
    ),
    "pleural_effusion": (
        "Pleural effusion detected. Assess for laterality, size, and "
        "clinical correlation with cardiac or infectious etiology."
    ),
    "cardiomegaly": (
        "Cardiomegaly identified (cardiothoracic ratio > 0.5). "
        "Echocardiographic correlation recommended."
    ),
    "fracture": (
        "Rib fracture(s) detected. Evaluate for associated pneumothorax "
        "or hemothorax. Pain management and respiratory monitoring advised."
    ),
}


class CXRRapidFindingsWorkflow(BaseImagingWorkflow):
    """Multi-label chest X-ray classification for emergency triage.

    Pipeline:
        1. Load CXR image (DICOM or PNG)
        2. Resize to 224x224 (DenseNet input), normalize with ImageNet stats
        3. Run DenseNet-121 multi-label classification
        4. Apply per-class confidence thresholds
        5. Optional: VILA-M3 generates natural language findings description
        6. Determine overall severity from highest-severity positive finding
    """

    WORKFLOW_NAME: str = "cxr_rapid_findings"
    TARGET_LATENCY_SEC: float = 30.0
    MODALITY: str = "cxr"
    BODY_REGION: str = "chest"
    MODELS_USED: List[str] = ["DenseNet-121 (MONAI)", "VILA-M3 (optional)"]

    def preprocess(self, input_path: str) -> Any:
        """Load chest X-ray and preprocess for DenseNet-121.

        Steps:
            - Load DICOM or PNG image
            - Convert to single-channel grayscale if needed
            - Resize to 224x224 pixels
            - Normalize with ImageNet statistics:
              mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            - Replicate single channel to 3 channels (DenseNet expects RGB)

        In production, uses monai.transforms:
            LoadImaged, EnsureChannelFirstd, Resized(224, 224),
            NormalizeIntensityd, RepeatChanneld(3)
        """
        logger.info(
            f"Preprocessing CXR from {input_path}: "
            "resize 224x224, ImageNet normalize, 3-channel"
        )
        return None

    def infer(self, preprocessed: Any) -> Dict:
        """Run DenseNet-121 multi-label classification.

        In production, runs the MONAI DenseNet-121 bundle trained on
        CheXpert/MIMIC-CXR for 5-class multi-label classification.
        Returns per-class sigmoid confidence scores.
        """
        if self.mock_mode:
            return self._mock_inference()

        logger.info("Running DenseNet-121 CXR classification via MONAI NIM")
        raise NotImplementedError(
            "Real inference requires MONAI DenseNet-121 NIM deployment. "
            "Set mock_mode=True for synthetic results."
        )

    def _mock_inference(self) -> Dict:
        """Return realistic mock multi-label CXR classification result.

        Simulates a case with consolidation (likely pneumonia) and small
        pleural effusion -- a common emergency department presentation.
        Pneumothorax, cardiomegaly, and fracture are below threshold.
        """
        return {
            "class_probabilities": {
                "pneumothorax": 0.08,
                "consolidation": 0.87,
                "pleural_effusion": 0.72,
                "cardiomegaly": 0.31,
                "fracture": 0.12,
            },
            "model_name": "DenseNet-121 (CheXpert)",
            "input_size": "224x224",
        }

    def postprocess(self, inference_result: Dict) -> WorkflowResult:
        """Apply per-class thresholds and determine overall severity.

        For each class, the sigmoid probability is compared against the
        class-specific threshold. Positive findings are added to the
        findings list. Overall severity is the maximum severity among
        all positive findings.
        """
        class_probs = inference_result.get("class_probabilities", {})

        if not class_probs:
            return WorkflowResult(
                workflow_name=self.WORKFLOW_NAME,
                status=WorkflowStatus.COMPLETED,
                findings=[{
                    "category": FindingCategory.NORMAL.value,
                    "description": "No classification output available",
                    "severity": FindingSeverity.ROUTINE.value,
                }],
                measurements={},
                classification="incomplete",
                severity=FindingSeverity.ROUTINE,
                nim_services_used=self.MODELS_USED,
            )

        findings = []
        positive_classes = []
        overall_severity = FindingSeverity.NORMAL

        severity_order = [
            FindingSeverity.NORMAL,
            FindingSeverity.ROUTINE,
            FindingSeverity.SIGNIFICANT,
            FindingSeverity.URGENT,
            FindingSeverity.CRITICAL,
        ]

        # Evaluate each class against its threshold
        for class_name, probability in class_probs.items():
            threshold = CXR_CLASS_THRESHOLDS.get(class_name, 0.50)
            is_positive = probability >= threshold

            if is_positive:
                positive_classes.append(class_name)
                class_severity = CXR_CLASS_SEVERITY.get(class_name, FindingSeverity.ROUTINE)
                class_category = CXR_CLASS_TO_CATEGORY.get(class_name, FindingCategory.NORMAL)
                class_description = CXR_CLASS_DESCRIPTIONS.get(class_name, "Finding detected.")

                findings.append({
                    "category": class_category.value,
                    "description": class_description,
                    "severity": class_severity.value,
                    "class_name": class_name,
                    "confidence": round(probability, 3),
                    "threshold": threshold,
                    "above_threshold": True,
                })

                if severity_order.index(class_severity) > severity_order.index(overall_severity):
                    overall_severity = class_severity

        # If no positive findings, report as normal
        if not findings:
            findings.append({
                "category": FindingCategory.NORMAL.value,
                "description": (
                    "No significant acute findings. All classification scores "
                    "below clinical thresholds."
                ),
                "severity": FindingSeverity.NORMAL.value,
            })

        # Build measurements from all class probabilities
        measurements = {}
        for class_name, probability in class_probs.items():
            measurements[f"{class_name}_confidence"] = round(probability, 4)

        # Classification summary
        if positive_classes:
            classification = f"positive: {', '.join(sorted(positive_classes))}"
        else:
            classification = "negative"

        return WorkflowResult(
            workflow_name=self.WORKFLOW_NAME,
            status=WorkflowStatus.COMPLETED,
            findings=findings,
            measurements=measurements,
            classification=classification,
            severity=overall_severity,
            nim_services_used=self.MODELS_USED,
        )
