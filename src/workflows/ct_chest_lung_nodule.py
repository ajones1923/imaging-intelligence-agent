"""CT Chest Lung Nodule Detection & Classification Workflow.

Reference workflow #2: Lung cancer screening nodule analysis.
Uses RetinaNet for detection + SegResNet for volumetric segmentation (MONAI).
Applies ACR Lung-RADS v2022 classification for standardized reporting.
"""

from typing import Any, Dict, List

from loguru import logger

from src.models import FindingCategory, FindingSeverity, LungRADS, WorkflowResult, WorkflowStatus
from src.workflows.base import BaseImagingWorkflow


# ACR Lung-RADS v2022 thresholds for solid nodules (mm)
LUNG_RADS_THRESHOLDS = {
    "cat_2_max": 6.0,    # Category 2: < 6mm solid
    "cat_3_max": 8.0,    # Category 3: 6-8mm solid
    "cat_4a_max": 15.0,  # Category 4A: 8-15mm solid
    # Category 4B: > 15mm solid
}


def classify_solid_nodule_lung_rads(diameter_mm: float) -> LungRADS:
    """Classify a solid nodule using ACR Lung-RADS v2022 size thresholds.

    Args:
        diameter_mm: Maximum axial diameter of the solid nodule in mm.

    Returns:
        LungRADS category enum value.

    Reference:
        ACR Lung-RADS v2022 Assessment Categories:
        - Category 2 (Benign Appearance): Solid nodule < 6mm
        - Category 3 (Probably Benign): Solid nodule 6-8mm
        - Category 4A (Suspicious): Solid nodule 8-15mm
        - Category 4B (Very Suspicious): Solid nodule >= 15mm
    """
    if diameter_mm < LUNG_RADS_THRESHOLDS["cat_2_max"]:
        return LungRADS.CAT_2
    elif diameter_mm < LUNG_RADS_THRESHOLDS["cat_3_max"]:
        return LungRADS.CAT_3
    elif diameter_mm < LUNG_RADS_THRESHOLDS["cat_4a_max"]:
        return LungRADS.CAT_4A
    else:
        return LungRADS.CAT_4B


def lung_rads_to_severity(category: LungRADS) -> FindingSeverity:
    """Map Lung-RADS category to clinical severity."""
    mapping = {
        LungRADS.CAT_1: FindingSeverity.NORMAL,
        LungRADS.CAT_2: FindingSeverity.ROUTINE,
        LungRADS.CAT_3: FindingSeverity.SIGNIFICANT,
        LungRADS.CAT_4A: FindingSeverity.URGENT,
        LungRADS.CAT_4B: FindingSeverity.CRITICAL,
        LungRADS.CAT_4X: FindingSeverity.CRITICAL,
    }
    return mapping.get(category, FindingSeverity.ROUTINE)


def lung_rads_recommendation(category: LungRADS) -> str:
    """Return ACR Lung-RADS v2022 management recommendation."""
    recommendations = {
        LungRADS.CAT_1: "Continue annual LDCT screening.",
        LungRADS.CAT_2: (
            "Continue annual LDCT screening. "
            "Solid nodule < 6mm — benign appearance."
        ),
        LungRADS.CAT_3: (
            "6-month LDCT follow-up recommended. "
            "Solid nodule 6-8mm — probably benign."
        ),
        LungRADS.CAT_4A: (
            "3-month LDCT follow-up or PET/CT recommended. "
            "Solid nodule 8-15mm — suspicious."
        ),
        LungRADS.CAT_4B: (
            "Chest CT with contrast, PET/CT, and/or tissue sampling recommended. "
            "Solid nodule >= 15mm — very suspicious."
        ),
        LungRADS.CAT_4X: (
            "Tissue sampling and/or multidisciplinary consultation recommended. "
            "Additional suspicious features present."
        ),
    }
    return recommendations.get(category, "Clinical correlation recommended.")


class CTChestLungNoduleWorkflow(BaseImagingWorkflow):
    """Lung nodule detection, segmentation, and Lung-RADS classification.

    Pipeline:
        1. Load DICOM CT chest series (low-dose or diagnostic)
        2. Resample to 1mm isotropic, apply lung window (-1000 to 400 HU)
        3. Run RetinaNet nodule detection (bounding boxes + confidence)
        4. Run SegResNet volumetric segmentation on detected nodules
        5. Measure diameter (mm), volume (mm3), density characteristics
        6. Apply ACR Lung-RADS v2022 classification per nodule
    """

    WORKFLOW_NAME: str = "ct_chest_lung_nodule"
    TARGET_LATENCY_SEC: float = 300.0
    MODALITY: str = "ct"
    BODY_REGION: str = "chest"
    MODELS_USED: List[str] = [
        "RetinaNet (MONAI)",
        "SegResNet (MONAI)",
        "VISTA-3D (optional)",
    ]

    def preprocess(self, input_path: str) -> Any:
        """Load DICOM CT chest and apply lung-optimized preprocessing.

        Steps:
            - Load DICOM series from input_path
            - Reorient to RAS coordinate system
            - Resample to 1mm isotropic voxel spacing
            - Apply lung window: center=-600 HU, width=1400 HU (range -1000 to 400 HU)
            - Normalize intensity to [0, 1]

        In production, uses monai.transforms:
            LoadImaged, EnsureChannelFirstd, Orientationd, Spacingd,
            ScaleIntensityRanged (a_min=-1000, a_max=400, b_min=0.0, b_max=1.0)
        """
        logger.info(
            f"Preprocessing CT chest from {input_path}: "
            "reorient RAS, resample 1mm iso, lung window -1000 to 400 HU"
        )
        return None

    def infer(self, preprocessed: Any) -> Dict:
        """Run nodule detection (RetinaNet) + segmentation (SegResNet).

        In production:
            1. RetinaNet detects candidate nodule regions with bounding boxes
            2. SegResNet segments each detected nodule for volumetric analysis
            3. Optional VISTA-3D provides anatomical context (lobe assignment)
        """
        if self.mock_mode:
            return self._mock_inference()

        logger.info("Running RetinaNet detection + SegResNet segmentation via MONAI NIM")
        raise NotImplementedError(
            "Real inference requires MONAI RetinaNet + SegResNet NIM deployment. "
            "Set mock_mode=True for synthetic results."
        )

    def _mock_inference(self) -> Dict:
        """Return realistic mock lung nodule detection result.

        Simulates a screening CT finding with two nodules:
            - A 7.2mm solid nodule in the right upper lobe (Lung-RADS 3)
            - A 12.8mm part-solid nodule in the left lower lobe (Lung-RADS 4A)
        """
        return {
            "nodules_detected": 2,
            "nodules": [
                {
                    "id": "nodule_1",
                    "location": "right upper lobe",
                    "lobe": "RUL",
                    "type": "solid",
                    "max_diameter_mm": 7.2,
                    "min_diameter_mm": 5.8,
                    "mean_diameter_mm": 6.5,
                    "volume_mm3": 195.0,
                    "mean_hu": -12.0,
                    "detection_confidence": 0.94,
                    "centroid_voxel": [128, 200, 310],
                    "bounding_box": [120, 192, 302, 136, 208, 318],
                    "segmentation_mask": None,
                    "morphology": "smooth",
                    "calcification": False,
                    "spiculation": False,
                },
                {
                    "id": "nodule_2",
                    "location": "left lower lobe",
                    "lobe": "LLL",
                    "type": "part-solid",
                    "max_diameter_mm": 12.8,
                    "min_diameter_mm": 10.1,
                    "mean_diameter_mm": 11.5,
                    "volume_mm3": 1102.0,
                    "solid_component_diameter_mm": 9.3,
                    "ground_glass_diameter_mm": 12.8,
                    "mean_hu": 24.0,
                    "detection_confidence": 0.97,
                    "centroid_voxel": [340, 280, 195],
                    "bounding_box": [328, 268, 183, 352, 292, 207],
                    "segmentation_mask": None,
                    "morphology": "lobulated",
                    "calcification": False,
                    "spiculation": True,
                },
            ],
        }

    def postprocess(self, inference_result: Dict) -> WorkflowResult:
        """Classify each nodule via ACR Lung-RADS v2022 and determine overall severity.

        For solid nodules, classification is based on maximum axial diameter.
        For part-solid nodules, the solid component diameter is used.
        Overall severity is driven by the highest-category nodule.
        """
        nodules = inference_result.get("nodules", [])
        nodules_detected = inference_result.get("nodules_detected", len(nodules))

        if nodules_detected == 0 or not nodules:
            return WorkflowResult(
                workflow_name=self.WORKFLOW_NAME,
                status=WorkflowStatus.COMPLETED,
                findings=[{
                    "category": FindingCategory.NORMAL.value,
                    "description": "No pulmonary nodules detected",
                    "severity": FindingSeverity.NORMAL.value,
                    "lung_rads": LungRADS.CAT_1.value,
                }],
                measurements={"nodule_count": 0},
                classification=f"Lung-RADS {LungRADS.CAT_1.value}",
                severity=FindingSeverity.NORMAL,
                nim_services_used=self.MODELS_USED,
            )

        findings = []
        overall_severity = FindingSeverity.NORMAL
        highest_lung_rads = LungRADS.CAT_1

        for nodule in nodules:
            nodule_id = nodule.get("id", "unknown")
            nodule_type = nodule.get("type", "solid")
            location = nodule.get("location", "unspecified")
            max_diameter = nodule.get("max_diameter_mm", 0.0)

            # For part-solid nodules, classify by solid component diameter
            if nodule_type == "part-solid":
                classify_diameter = nodule.get("solid_component_diameter_mm", max_diameter)
            else:
                classify_diameter = max_diameter

            lung_rads = classify_solid_nodule_lung_rads(classify_diameter)
            nodule_severity = lung_rads_to_severity(lung_rads)
            recommendation = lung_rads_recommendation(lung_rads)

            # Track overall severity (highest wins)
            severity_order = [
                FindingSeverity.NORMAL,
                FindingSeverity.ROUTINE,
                FindingSeverity.SIGNIFICANT,
                FindingSeverity.URGENT,
                FindingSeverity.CRITICAL,
            ]
            if severity_order.index(nodule_severity) > severity_order.index(overall_severity):
                overall_severity = nodule_severity
                highest_lung_rads = lung_rads

            # Build per-nodule measurement dict
            nodule_measurements = {
                "max_diameter_mm": max_diameter,
                "volume_mm3": nodule.get("volume_mm3", 0.0),
                "detection_confidence": nodule.get("detection_confidence", 0.0),
            }
            if nodule_type == "part-solid":
                nodule_measurements["solid_component_mm"] = nodule.get(
                    "solid_component_diameter_mm", 0.0
                )
                nodule_measurements["ground_glass_mm"] = nodule.get(
                    "ground_glass_diameter_mm", 0.0
                )

            morphology_notes = []
            if nodule.get("spiculation"):
                morphology_notes.append("spiculated")
            if nodule.get("calcification"):
                morphology_notes.append("calcified")
            if nodule.get("morphology"):
                morphology_notes.append(nodule["morphology"])

            findings.append({
                "category": FindingCategory.NODULE.value,
                "description": (
                    f"{nodule_type.replace('-', ' ').title()} nodule in {location}, "
                    f"{max_diameter:.1f}mm max diameter, "
                    f"Lung-RADS {lung_rads.value}"
                ),
                "severity": nodule_severity.value,
                "nodule_id": nodule_id,
                "nodule_type": nodule_type,
                "location": location,
                "lobe": nodule.get("lobe", ""),
                "lung_rads": lung_rads.value,
                "recommendation": recommendation,
                "measurements": nodule_measurements,
                "morphology": ", ".join(morphology_notes) if morphology_notes else "unremarkable",
            })

        # Aggregate measurements
        aggregate_measurements = {
            "nodule_count": float(nodules_detected),
        }
        for i, nodule in enumerate(nodules):
            prefix = f"nodule_{i + 1}"
            aggregate_measurements[f"{prefix}_diameter_mm"] = nodule.get("max_diameter_mm", 0.0)
            aggregate_measurements[f"{prefix}_volume_mm3"] = nodule.get("volume_mm3", 0.0)

        return WorkflowResult(
            workflow_name=self.WORKFLOW_NAME,
            status=WorkflowStatus.COMPLETED,
            findings=findings,
            measurements=aggregate_measurements,
            classification=f"Lung-RADS {highest_lung_rads.value}",
            severity=overall_severity,
            nim_services_used=self.MODELS_USED,
        )
