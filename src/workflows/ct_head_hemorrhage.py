"""CT Head Hemorrhage Detection & Segmentation Workflow.

Reference workflow #1: Emergency intracranial hemorrhage triage.
Uses 3D U-Net (MONAI) for hemorrhage segmentation with optional VISTA-3D.
Applies Brain Trauma Foundation (BTF) severity thresholds for clinical triage.
"""

from typing import Any, Dict, List

from loguru import logger

from src.models import FindingCategory, FindingSeverity, WorkflowResult, WorkflowStatus
from src.workflows.base import BaseImagingWorkflow


class CTHeadHemorrhageWorkflow(BaseImagingWorkflow):
    """Intracranial hemorrhage detection, segmentation, and triage.

    Pipeline:
        1. Load DICOM CT head series
        2. Reorient to RAS, resample 1mm isotropic, apply CT blood window (0-80 HU)
        3. Run 3D U-Net hemorrhage segmentation
        4. Measure volume (mL), midline shift (mm), max thickness (mm)
        5. Apply Brain Trauma Foundation thresholds for severity
    """

    WORKFLOW_NAME: str = "ct_head_hemorrhage"
    TARGET_LATENCY_SEC: float = 90.0
    MODALITY: str = "ct"
    BODY_REGION: str = "head"
    MODELS_USED: List[str] = ["3D U-Net (MONAI)", "VISTA-3D (optional)"]

    # Brain Trauma Foundation thresholds
    BTF_VOLUME_CRITICAL_ML: float = 30.0
    BTF_SHIFT_CRITICAL_MM: float = 5.0
    BTF_THICKNESS_CRITICAL_MM: float = 10.0
    BTF_VOLUME_URGENT_ML: float = 5.0

    HEMORRHAGE_TYPES = [
        "epidural",
        "subdural",
        "subarachnoid",
        "intraparenchymal",
        "intraventricular",
    ]

    def preprocess(self, input_path: str) -> Any:
        """Load DICOM CT head and apply standard preprocessing.

        Steps:
            - Load DICOM series from input_path
            - Reorient to RAS (Right-Anterior-Superior) coordinate system
            - Resample to 1mm isotropic voxel spacing
            - Apply CT blood window: center=40 HU, width=80 HU (range 0-80 HU)
            - Normalize intensity to [0, 1]

        In production, uses monai.transforms:
            LoadImaged, EnsureChannelFirstd, Orientationd, Spacingd,
            ScaleIntensityRanged (a_min=0, a_max=80, b_min=0.0, b_max=1.0)
        """
        logger.info(
            f"Preprocessing CT head from {input_path}: "
            "reorient RAS, resample 1mm iso, blood window 0-80 HU"
        )
        # In real mode, would return a preprocessed tensor via MONAI transforms
        return None

    def infer(self, preprocessed: Any) -> Dict:
        """Run 3D U-Net hemorrhage segmentation.

        In production, loads the MONAI Bundle checkpoint and runs inference
        on the preprocessed volume. Returns segmentation mask + measurements.
        """
        if self.mock_mode:
            return self._mock_inference()

        logger.info("Running 3D U-Net hemorrhage segmentation via MONAI NIM")
        # In production:
        #   client = self.nim_clients.get("monai_3d_unet")
        #   result = client.infer(preprocessed)
        #   mask = result["segmentation_mask"]
        #   measurements = self._compute_measurements(mask)
        #   return {"segmentation_mask": mask, **measurements}
        raise NotImplementedError(
            "Real inference requires MONAI 3D U-Net NIM deployment. "
            "Set mock_mode=True for synthetic results."
        )

    def _mock_inference(self) -> Dict:
        """Return realistic mock hemorrhage detection result.

        Simulates a moderate intraparenchymal hemorrhage in the right
        basal ganglia region -- a common hypertensive hemorrhage presentation.
        """
        return {
            "hemorrhage_detected": True,
            "hemorrhage_type": "intraparenchymal",
            "location": "right basal ganglia",
            "volume_ml": 12.5,
            "midline_shift_mm": 3.2,
            "max_thickness_mm": 8.1,
            "hounsfield_mean": 62.0,
            "hounsfield_max": 78.0,
            "segmentation_mask": None,  # Would be numpy array in real mode
            "surrounding_edema_ml": 4.3,
            "intraventricular_extension": False,
        }

    def postprocess(self, inference_result: Dict) -> WorkflowResult:
        """Apply Brain Trauma Foundation severity thresholds.

        Severity classification (BTF Guidelines, 4th Edition):
            CRITICAL: volume > 30 mL OR midline shift > 5 mm OR thickness > 10 mm
            URGENT:   volume > 5 mL (but below critical thresholds)
            ROUTINE:  hemorrhage detected but below urgent thresholds
            NORMAL:   no hemorrhage detected
        """
        hemorrhage_detected = inference_result.get("hemorrhage_detected", False)

        if not hemorrhage_detected:
            return WorkflowResult(
                workflow_name=self.WORKFLOW_NAME,
                status=WorkflowStatus.COMPLETED,
                findings=[{
                    "category": FindingCategory.NORMAL.value,
                    "description": "No intracranial hemorrhage detected",
                    "severity": FindingSeverity.NORMAL.value,
                }],
                measurements={},
                classification="no_hemorrhage",
                severity=FindingSeverity.NORMAL,
                nim_services_used=self.MODELS_USED,
            )

        volume_ml = inference_result.get("volume_ml", 0.0)
        midline_shift_mm = inference_result.get("midline_shift_mm", 0.0)
        max_thickness_mm = inference_result.get("max_thickness_mm", 0.0)
        hemorrhage_type = inference_result.get("hemorrhage_type", "unspecified")
        location = inference_result.get("location", "unspecified")

        # Brain Trauma Foundation severity thresholds
        if (
            volume_ml > self.BTF_VOLUME_CRITICAL_ML
            or midline_shift_mm > self.BTF_SHIFT_CRITICAL_MM
            or max_thickness_mm > self.BTF_THICKNESS_CRITICAL_MM
        ):
            severity = FindingSeverity.CRITICAL
            classification = "critical_hemorrhage"
            recommendation = (
                "STAT neurosurgical consultation. Consider emergent surgical "
                "evacuation per Brain Trauma Foundation guidelines."
            )
        elif volume_ml > self.BTF_VOLUME_URGENT_ML:
            severity = FindingSeverity.URGENT
            classification = "urgent_hemorrhage"
            recommendation = (
                "Urgent neurosurgical consultation recommended. Serial CT imaging "
                "in 6-8 hours to assess for expansion."
            )
        else:
            severity = FindingSeverity.ROUTINE
            classification = "small_hemorrhage"
            recommendation = (
                "Neurology consultation. Repeat CT in 24 hours. "
                "Monitor neurological status."
            )

        measurements = {
            "volume_ml": volume_ml,
            "midline_shift_mm": midline_shift_mm,
            "max_thickness_mm": max_thickness_mm,
        }

        if "hounsfield_mean" in inference_result:
            measurements["hounsfield_mean"] = inference_result["hounsfield_mean"]
        if "hounsfield_max" in inference_result:
            measurements["hounsfield_max"] = inference_result["hounsfield_max"]
        if "surrounding_edema_ml" in inference_result:
            measurements["surrounding_edema_ml"] = inference_result["surrounding_edema_ml"]

        findings = [
            {
                "category": FindingCategory.HEMORRHAGE.value,
                "description": (
                    f"{hemorrhage_type.replace('_', ' ').title()} hemorrhage "
                    f"in {location}, volume {volume_ml:.1f} mL, "
                    f"midline shift {midline_shift_mm:.1f} mm, "
                    f"max thickness {max_thickness_mm:.1f} mm"
                ),
                "severity": severity.value,
                "hemorrhage_type": hemorrhage_type,
                "location": location,
                "recommendation": recommendation,
                "btf_thresholds": {
                    "volume_critical_ml": self.BTF_VOLUME_CRITICAL_ML,
                    "shift_critical_mm": self.BTF_SHIFT_CRITICAL_MM,
                    "thickness_critical_mm": self.BTF_THICKNESS_CRITICAL_MM,
                },
            }
        ]

        # Add intraventricular extension as separate finding if present
        if inference_result.get("intraventricular_extension", False):
            findings.append({
                "category": FindingCategory.HEMORRHAGE.value,
                "description": "Intraventricular extension of hemorrhage detected",
                "severity": FindingSeverity.URGENT.value,
                "hemorrhage_type": "intraventricular",
            })

        return WorkflowResult(
            workflow_name=self.WORKFLOW_NAME,
            status=WorkflowStatus.COMPLETED,
            findings=findings,
            measurements=measurements,
            classification=classification,
            severity=severity,
            nim_services_used=self.MODELS_USED,
        )
