"""VILA-M3 NIM client for medical visual language understanding.

VILA-M3 (Visual Language Model for Medical 3D) provides visual question
answering, report generation, and zero-shot classification on medical
images using a Llama-3 backbone with visual encoding.
"""

import base64
import random
from pathlib import Path
from typing import Any, Dict, List, Optional

from loguru import logger

from src.models import VLMResponse

from .base import BaseNIMClient


class VILAM3Client(BaseNIMClient):
    """Client for NVIDIA VILA-M3 NIM visual language model service.

    VILA-M3 uses an OpenAI-compatible API format with a Llama-3
    backbone for multimodal medical image understanding. Capabilities:
      - Visual question answering on radiology images
      - Structured radiology report generation
      - Zero-shot image classification
    """

    def __init__(self, base_url: str, mock_enabled: bool = True):
        super().__init__(base_url, service_name="vila_m3", mock_enabled=mock_enabled)

    def _encode_image(self, image_path: str) -> str:
        """Read and base64-encode an image file."""
        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    def _build_vlm_payload(
        self,
        image_path: str,
        prompt: str,
        temperature: float = 0.1,
        max_tokens: int = 1024,
    ) -> Dict[str, Any]:
        """Build an OpenAI-compatible chat completion payload with image."""
        image_b64 = self._encode_image(image_path)

        # Determine MIME type from extension
        ext = Path(image_path).suffix.lower()
        mime_map = {
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".dcm": "application/dicom",
            ".nii": "application/octet-stream",
            ".nii.gz": "application/octet-stream",
        }
        mime_type = mime_map.get(ext, "image/png")

        return {
            "model": "vila-m3",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{mime_type};base64,{image_b64}"
                            },
                        },
                        {
                            "type": "text",
                            "text": prompt,
                        },
                    ],
                }
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

    def analyze_image(
        self,
        image_path: str,
        question: str,
    ) -> VLMResponse:
        """Visual question answering on a medical image.

        Args:
            image_path: Path to the medical image (PNG, JPEG, DICOM, NIfTI).
            question: Clinical question about the image.

        Returns:
            VLMResponse with answer text, confidence, and metadata.
        """
        logger.info(f"VQA on {image_path}: {question[:80]}...")

        # Build payload only when not using mock (avoids file-not-found in mock mode)
        if self.is_available():
            payload = self._build_vlm_payload(image_path, question)
        else:
            payload = {}

        result = self._invoke_or_mock(
            endpoint="/v1/chat/completions",
            payload=payload,
            timeout=120,
            image_path=image_path,
            question=question,
            mode="vqa",
        )

        # Parse OpenAI-compatible response into VLMResponse
        if isinstance(result, dict) and not isinstance(result, VLMResponse):
            answer = ""
            if "choices" in result and len(result["choices"]) > 0:
                answer = result["choices"][0].get("message", {}).get("content", "")

            return VLMResponse(
                answer=answer,
                question=question,
                image_path=image_path,
                confidence=result.get("confidence", 0.0),
                model_name="vila-m3",
                is_mock=False,
            )

        return result

    def generate_report(
        self,
        image_path: str,
        findings_context: str = "",
    ) -> str:
        """Generate a structured radiology report from a medical image.

        Args:
            image_path: Path to the medical image.
            findings_context: Optional prior findings or clinical context
                to incorporate into the report.

        Returns:
            Structured radiology report as a string.
        """
        logger.info(f"Generating radiology report for: {image_path}")

        prompt = (
            "You are an expert radiologist. Generate a structured radiology report "
            "for this medical image. Include the following sections:\n"
            "1. TECHNIQUE\n"
            "2. COMPARISON\n"
            "3. FINDINGS\n"
            "4. IMPRESSION\n\n"
            "Be specific about anatomical structures, measurements, and any "
            "abnormalities observed."
        )
        if findings_context:
            prompt += f"\n\nAdditional clinical context:\n{findings_context}"

        if self.is_available():
            payload = self._build_vlm_payload(
                image_path, prompt, temperature=0.1, max_tokens=2048
            )
        else:
            payload = {}

        result = self._invoke_or_mock(
            endpoint="/v1/chat/completions",
            payload=payload,
            timeout=180,
            image_path=image_path,
            findings_context=findings_context,
            mode="report",
        )

        # Extract report text
        if isinstance(result, dict):
            if "choices" in result and len(result["choices"]) > 0:
                return result["choices"][0].get("message", {}).get("content", "")
            return result.get("report", "")
        if isinstance(result, VLMResponse):
            return result.answer

        return str(result)

    def classify(
        self,
        image_path: str,
        labels: List[str],
    ) -> Dict[str, float]:
        """Zero-shot classification of a medical image.

        Args:
            image_path: Path to the medical image.
            labels: List of candidate classification labels.

        Returns:
            Dictionary mapping each label to its probability score.
        """
        logger.info(
            f"Zero-shot classification with {len(labels)} labels: {image_path}"
        )

        labels_str = ", ".join(f'"{lbl}"' for lbl in labels)
        prompt = (
            f"Classify this medical image into exactly one of the following "
            f"categories: [{labels_str}]. For each category, provide a "
            f"confidence score between 0.0 and 1.0. Respond in JSON format "
            f'like: {{"label1": 0.85, "label2": 0.10, ...}}'
        )

        if self.is_available():
            payload = self._build_vlm_payload(
                image_path, prompt, temperature=0.0, max_tokens=512
            )
        else:
            payload = {}

        result = self._invoke_or_mock(
            endpoint="/v1/chat/completions",
            payload=payload,
            timeout=120,
            image_path=image_path,
            labels=labels,
            mode="classify",
        )

        # Parse classification scores from response
        content = ""
        if isinstance(result, VLMResponse):
            content = result.answer
        elif isinstance(result, dict):
            if "choices" in result and len(result["choices"]) > 0:
                content = result["choices"][0].get("message", {}).get("content", "")
            elif all(lbl in result for lbl in labels):
                return {lbl: float(result[lbl]) for lbl in labels}

        if content:
            try:
                import json
                scores = json.loads(content)
                if isinstance(scores, dict):
                    return {k: float(v) for k, v in scores.items()}
            except (json.JSONDecodeError, ValueError):
                logger.warning(
                    f"Could not parse classification JSON from VLM: {content[:200]}"
                )

        # Fallback: return uniform scores
        logger.warning("Could not extract classification scores, returning uniform")
        uniform = 1.0 / len(labels) if labels else 0.0
        return {lbl: uniform for lbl in labels}

    def _mock_response(self, **kwargs) -> VLMResponse:
        """Return a realistic mock VLMResponse.

        Generates plausible radiology findings based on the mode
        (VQA, report generation, or classification).
        """
        mode = kwargs.get("mode", "vqa")
        image_path = kwargs.get("image_path", "unknown.png")
        question = kwargs.get("question", "")

        if mode == "report":
            answer = (
                "TECHNIQUE: CT of the chest without contrast.\n\n"
                "COMPARISON: No prior studies available.\n\n"
                "FINDINGS:\n"
                "- Lungs: Clear bilaterally. No focal consolidation, "
                "pleural effusion, or pneumothorax.\n"
                "- Heart: Normal size. No pericardial effusion.\n"
                "- Mediastinum: No lymphadenopathy. Normal vascular structures.\n"
                "- Bones: No acute fracture or destructive lesion.\n"
                "- Upper abdomen: Limited evaluation; visualized portions "
                "appear unremarkable.\n\n"
                "IMPRESSION:\n"
                "1. No acute cardiopulmonary abnormality.\n"
                "2. Normal chest CT examination."
            )
        elif mode == "classify":
            labels = kwargs.get("labels", ["normal", "abnormal"])
            # Generate mock classification scores
            import json
            primary_idx = random.randint(0, len(labels) - 1)
            scores = {}
            remaining = 1.0
            for i, lbl in enumerate(labels):
                if i == primary_idx:
                    scores[lbl] = round(random.uniform(0.65, 0.90), 3)
                    remaining -= scores[lbl]
                else:
                    score = round(remaining / (len(labels) - i), 3)
                    scores[lbl] = max(0.01, score)
                    remaining -= scores[lbl]

            answer = json.dumps(scores, indent=2)
        else:
            # VQA mode
            findings = [
                "The image shows normal lung parenchyma with no evidence "
                "of consolidation or ground-glass opacities.",
                "The cardiac silhouette is within normal limits.",
                "No pleural effusion or pneumothorax is identified.",
                "The mediastinal structures appear normal.",
                "The visualized osseous structures are intact without "
                "acute fracture.",
            ]
            selected = random.sample(findings, min(3, len(findings)))
            answer = " ".join(selected)

        logger.info(f"Mock VILA-M3 response generated ({mode} mode)")

        return VLMResponse(
            answer=answer,
            question=question,
            image_path=image_path,
            confidence=round(random.uniform(0.70, 0.95), 3),
            model_name="vila-m3-mock",
            is_mock=True,
        )
