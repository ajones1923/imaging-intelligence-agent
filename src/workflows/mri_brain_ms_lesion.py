"""MRI Brain MS Lesion Segmentation & Disease Activity Workflow.

Reference workflow #4: Multiple Sclerosis lesion quantification.
Uses 3D U-Net (MONAI) for white matter lesion segmentation with optional VISTA-3D.
Classifies disease activity based on new/enlarging lesion counts per MAGNIMS criteria.
"""

from typing import Any, Dict, List

from loguru import logger

from src.models import FindingCategory, FindingSeverity, WorkflowResult, WorkflowStatus
from src.workflows.base import BaseImagingWorkflow


# Disease activity classification thresholds (MAGNIMS consensus guidelines)
MS_ACTIVITY_THRESHOLDS = {
    "highly_active_new_lesion_count": 3,  # > 3 new lesions = highly active
    "active_new_lesion_count": 0,          # > 0 new lesions = active
    # 0 new lesions = stable
}


def classify_ms_activity(new_lesion_count: int) -> str:
    """Classify MS disease activity based on new/enlarging lesion count.

    Based on MAGNIMS/EMA consensus:
        - Highly active: > 3 new or unequivocally enlarging T2 lesions
        - Active: 1-3 new or unequivocally enlarging T2 lesions
        - Stable: No new or enlarging lesions

    Args:
        new_lesion_count: Number of new or unequivocally enlarging T2 lesions.

    Returns:
        Activity classification string.
    """
    if new_lesion_count > MS_ACTIVITY_THRESHOLDS["highly_active_new_lesion_count"]:
        return "highly_active"
    elif new_lesion_count > MS_ACTIVITY_THRESHOLDS["active_new_lesion_count"]:
        return "active"
    else:
        return "stable"


def ms_activity_to_severity(activity: str) -> FindingSeverity:
    """Map MS disease activity classification to clinical severity."""
    mapping = {
        "highly_active": FindingSeverity.URGENT,
        "active": FindingSeverity.SIGNIFICANT,
        "stable": FindingSeverity.ROUTINE,
    }
    return mapping.get(activity, FindingSeverity.ROUTINE)


def ms_activity_recommendation(activity: str) -> str:
    """Return clinical recommendation based on MS disease activity."""
    recommendations = {
        "highly_active": (
            "Highly active disease. Consider escalation to higher-efficacy DMT "
            "(e.g., natalizumab, ocrelizumab, alemtuzumab). Urgent neurology review. "
            "Assess for treatment non-compliance or suboptimal response."
        ),
        "active": (
            "Active disease with new lesions detected. Discuss DMT optimization "
            "with treating neurologist. Follow-up MRI in 3-6 months to assess "
            "treatment response."
        ),
        "stable": (
            "Stable disease. No new or enlarging lesions identified. Continue "
            "current DMT regimen. Routine follow-up MRI in 12 months per "
            "MAGNIMS guidelines."
        ),
    }
    return recommendations.get(activity, "Clinical correlation recommended.")


class MRIBrainMSLesionWorkflow(BaseImagingWorkflow):
    """MS lesion segmentation, quantification, and disease activity classification.

    Pipeline:
        1. Load DICOM MRI brain series (FLAIR + optional T1 post-contrast)
        2. Skull strip, register to MNI template, normalize intensities
        3. Run 3D U-Net lesion segmentation on FLAIR sequence
        4. Per-lesion analysis: centroid, volume, location
        5. Compare with prior study if available (new vs. stable lesions)
        6. Classify disease activity per MAGNIMS consensus guidelines
    """

    WORKFLOW_NAME: str = "mri_brain_ms_lesion"
    TARGET_LATENCY_SEC: float = 300.0
    MODALITY: str = "mri"
    BODY_REGION: str = "brain"
    MODELS_USED: List[str] = ["3D U-Net (MONAI)", "VISTA-3D (optional)"]

    def preprocess(self, input_path: str) -> Any:
        """Load DICOM MRI brain and apply MS-specific preprocessing.

        Steps:
            - Load FLAIR DICOM series from input_path
            - Skull stripping (BET or SynthStrip)
            - Register to MNI152 template (affine + optional non-linear)
            - Resample to 1mm isotropic voxel spacing
            - Z-score intensity normalization within brain mask
            - Optional: co-register T1 post-contrast for enhancing lesion detection

        In production, uses monai.transforms:
            LoadImaged, EnsureChannelFirstd, Orientationd, Spacingd,
            NormalizeIntensityd (z-score within mask)
        """
        logger.info(
            f"Preprocessing MRI brain from {input_path}: "
            "skull strip, MNI registration, resample 1mm iso, z-score normalize"
        )
        return None

    def infer(self, preprocessed: Any) -> Dict:
        """Run 3D U-Net white matter lesion segmentation.

        In production, loads the MONAI MS lesion segmentation bundle and
        runs inference on the preprocessed FLAIR volume. Outputs a binary
        lesion mask from which per-lesion statistics are computed.
        """
        if self.mock_mode:
            return self._mock_inference()

        logger.info("Running 3D U-Net MS lesion segmentation via MONAI NIM")
        raise NotImplementedError(
            "Real inference requires MONAI 3D U-Net MS lesion NIM deployment. "
            "Set mock_mode=True for synthetic results."
        )

    def _mock_inference(self) -> Dict:
        """Return realistic mock MS lesion segmentation result.

        Simulates a relapsing-remitting MS patient with:
            - 14 total T2/FLAIR lesions (typical moderate lesion burden)
            - 2 new lesions compared to prior study (active disease)
            - Periventricular and juxtacortical distribution (classic MS pattern)
        """
        return {
            "lesion_count": 14,
            "total_lesion_volume_ml": 8.7,
            "new_lesion_count": 2,
            "enlarging_lesion_count": 1,
            "enhancing_lesion_count": 1,
            "brain_parenchymal_fraction": 0.82,
            "lesions": [
                {
                    "id": "lesion_01",
                    "centroid_mni": [28.5, -12.3, 18.7],
                    "volume_ml": 1.2,
                    "location": "periventricular right",
                    "is_new": False,
                    "is_enhancing": False,
                    "max_diameter_mm": 14.2,
                },
                {
                    "id": "lesion_02",
                    "centroid_mni": [-22.1, 8.4, 25.1],
                    "volume_ml": 0.9,
                    "location": "periventricular left",
                    "is_new": False,
                    "is_enhancing": False,
                    "max_diameter_mm": 12.8,
                },
                {
                    "id": "lesion_03",
                    "centroid_mni": [15.3, 22.7, 32.0],
                    "volume_ml": 0.4,
                    "location": "juxtacortical right frontal",
                    "is_new": True,
                    "is_enhancing": True,
                    "max_diameter_mm": 8.5,
                },
                {
                    "id": "lesion_04",
                    "centroid_mni": [-8.2, -28.5, 14.3],
                    "volume_ml": 0.6,
                    "location": "periventricular left occipital",
                    "is_new": False,
                    "is_enhancing": False,
                    "max_diameter_mm": 10.1,
                },
                {
                    "id": "lesion_05",
                    "centroid_mni": [5.1, 15.8, 40.2],
                    "volume_ml": 0.3,
                    "location": "juxtacortical left parietal",
                    "is_new": True,
                    "is_enhancing": False,
                    "max_diameter_mm": 7.2,
                },
                {
                    "id": "lesion_06",
                    "centroid_mni": [18.9, -5.6, 22.4],
                    "volume_ml": 0.8,
                    "location": "periventricular right",
                    "is_new": False,
                    "is_enhancing": False,
                    "max_diameter_mm": 11.3,
                },
                {
                    "id": "lesion_07",
                    "centroid_mni": [-14.7, -18.2, 28.9],
                    "volume_ml": 0.5,
                    "location": "periventricular left",
                    "is_new": False,
                    "is_enhancing": False,
                    "max_diameter_mm": 9.0,
                },
                {
                    "id": "lesion_08",
                    "centroid_mni": [32.1, 5.3, 15.6],
                    "volume_ml": 0.7,
                    "location": "deep white matter right",
                    "is_new": False,
                    "is_enhancing": False,
                    "max_diameter_mm": 10.5,
                },
                {
                    "id": "lesion_09",
                    "centroid_mni": [-25.8, 12.1, 19.3],
                    "volume_ml": 0.4,
                    "location": "deep white matter left",
                    "is_new": False,
                    "is_enhancing": False,
                    "max_diameter_mm": 8.2,
                },
                {
                    "id": "lesion_10",
                    "centroid_mni": [10.2, -32.4, 20.1],
                    "volume_ml": 0.6,
                    "location": "periventricular right occipital",
                    "is_new": False,
                    "is_enhancing": False,
                    "max_diameter_mm": 9.8,
                },
                {
                    "id": "lesion_11",
                    "centroid_mni": [-5.4, 30.2, 35.7],
                    "volume_ml": 0.3,
                    "location": "juxtacortical left frontal",
                    "is_new": False,
                    "is_enhancing": False,
                    "max_diameter_mm": 7.5,
                },
                {
                    "id": "lesion_12",
                    "centroid_mni": [22.6, -8.9, 30.4],
                    "volume_ml": 0.5,
                    "location": "corpus callosum",
                    "is_new": False,
                    "is_enhancing": False,
                    "max_diameter_mm": 9.3,
                },
                {
                    "id": "lesion_13",
                    "centroid_mni": [-18.3, 25.6, 22.8],
                    "volume_ml": 0.8,
                    "location": "periventricular left frontal",
                    "is_new": False,
                    "is_enhancing": False,
                    "max_diameter_mm": 11.0,
                },
                {
                    "id": "lesion_14",
                    "centroid_mni": [7.8, -15.2, 38.5],
                    "volume_ml": 0.7,
                    "location": "deep white matter bilateral",
                    "is_new": False,
                    "is_enhancing": False,
                    "max_diameter_mm": 10.2,
                },
            ],
            "segmentation_mask": None,  # Would be numpy array in real mode
        }

    def postprocess(self, inference_result: Dict) -> WorkflowResult:
        """Classify MS disease activity and generate structured findings.

        Disease activity classification (MAGNIMS/EMA consensus):
            highly_active: > 3 new or enlarging T2 lesions -> URGENT
            active:        1-3 new or enlarging T2 lesions -> SIGNIFICANT
            stable:        0 new or enlarging lesions      -> ROUTINE
        """
        lesion_count = inference_result.get("lesion_count", 0)
        total_volume_ml = inference_result.get("total_lesion_volume_ml", 0.0)
        new_lesion_count = inference_result.get("new_lesion_count", 0)
        enlarging_count = inference_result.get("enlarging_lesion_count", 0)
        enhancing_count = inference_result.get("enhancing_lesion_count", 0)
        lesions = inference_result.get("lesions", [])
        brain_parenchymal_fraction = inference_result.get("brain_parenchymal_fraction", 0.0)

        # Combine new + enlarging for activity classification
        active_lesion_count = new_lesion_count + enlarging_count
        activity = classify_ms_activity(active_lesion_count)
        severity = ms_activity_to_severity(activity)
        recommendation = ms_activity_recommendation(activity)

        # Build findings list
        findings = []

        # Primary finding: disease activity summary
        findings.append({
            "category": FindingCategory.LESION.value,
            "description": (
                f"Multiple sclerosis: {lesion_count} total T2/FLAIR lesions, "
                f"total volume {total_volume_ml:.1f} mL. "
                f"{new_lesion_count} new, {enlarging_count} enlarging, "
                f"{enhancing_count} enhancing lesion(s). "
                f"Disease activity: {activity.replace('_', ' ')}."
            ),
            "severity": severity.value,
            "disease_activity": activity,
            "recommendation": recommendation,
        })

        # Per-lesion findings for new/enhancing lesions (clinically significant)
        for lesion in lesions:
            if lesion.get("is_new") or lesion.get("is_enhancing"):
                lesion_notes = []
                if lesion.get("is_new"):
                    lesion_notes.append("NEW")
                if lesion.get("is_enhancing"):
                    lesion_notes.append("ENHANCING")

                findings.append({
                    "category": FindingCategory.LESION.value,
                    "description": (
                        f"[{'/'.join(lesion_notes)}] Lesion {lesion['id']} in "
                        f"{lesion.get('location', 'unspecified')}, "
                        f"volume {lesion.get('volume_ml', 0.0):.2f} mL, "
                        f"max diameter {lesion.get('max_diameter_mm', 0.0):.1f} mm"
                    ),
                    "severity": FindingSeverity.SIGNIFICANT.value,
                    "lesion_id": lesion["id"],
                    "location": lesion.get("location", ""),
                    "volume_ml": lesion.get("volume_ml", 0.0),
                    "max_diameter_mm": lesion.get("max_diameter_mm", 0.0),
                    "centroid_mni": lesion.get("centroid_mni", []),
                    "is_new": lesion.get("is_new", False),
                    "is_enhancing": lesion.get("is_enhancing", False),
                })

        # Build aggregate measurements
        measurements = {
            "total_lesion_count": float(lesion_count),
            "total_lesion_volume_ml": total_volume_ml,
            "new_lesion_count": float(new_lesion_count),
            "enlarging_lesion_count": float(enlarging_count),
            "enhancing_lesion_count": float(enhancing_count),
            "active_lesion_count": float(active_lesion_count),
        }

        if brain_parenchymal_fraction > 0:
            measurements["brain_parenchymal_fraction"] = brain_parenchymal_fraction

        # Per-lesion volumes for trending
        for lesion in lesions:
            lid = lesion.get("id", "unknown")
            measurements[f"{lid}_volume_ml"] = lesion.get("volume_ml", 0.0)

        return WorkflowResult(
            workflow_name=self.WORKFLOW_NAME,
            status=WorkflowStatus.COMPLETED,
            findings=findings,
            measurements=measurements,
            classification=f"ms_{activity}",
            severity=severity,
            nim_services_used=self.MODELS_USED,
        )
