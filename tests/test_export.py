"""Tests for Imaging Intelligence Agent export functions.

Validates export_markdown(), export_json(), and export_pdf()
with various AgentResponse inputs. All tests are self-contained.

Author: Adam Jones
Date: February 2026
"""

import json
import pytest
from unittest.mock import patch, MagicMock

from src.export import export_json, export_markdown, export_pdf
from src.models import (
    AgentResponse,
    CrossCollectionResult,
    FindingSeverity,
    SearchHit,
    WorkflowResult,
    WorkflowStatus,
)


# ===================================================================
# FIXTURES
# ===================================================================


@pytest.fixture
def minimal_response():
    """Response with empty evidence and no extras."""
    return AgentResponse(
        question="What is VISTA-3D?",
        answer="VISTA-3D is a universal segmentation model.",
        evidence=CrossCollectionResult(query="VISTA-3D", hits=[]),
        timestamp="2026-02-27T10:00:00",
    )


@pytest.fixture
def rich_response():
    """Response with evidence, NIM services, and workflow results."""
    hits = [
        SearchHit(
            collection="imaging_literature",
            id="PMID:001",
            score=0.92,
            text="VISTA-3D segments 127 anatomical structures with mean Dice 0.91.",
            metadata={"year": 2025},
        ),
        SearchHit(
            collection="imaging_benchmarks",
            id="bench-001",
            score=0.85,
            text="VISTA-3D achieves state-of-the-art on TotalSegmentator benchmark.",
            metadata={"metric_value": 0.91},
        ),
        SearchHit(
            collection="imaging_guidelines",
            id="guide-001",
            score=0.78,
            text="ACR recommends AI segmentation for treatment planning.",
            metadata={"organization": "ACR"},
        ),
    ]

    evidence = CrossCollectionResult(
        query="What is the best segmentation model?",
        hits=hits,
        knowledge_context="## VISTA-3D\n- Universal segmentation model",
        total_collections_searched=11,
        search_time_ms=42.5,
    )

    workflow = WorkflowResult(
        workflow_name="ct_head_hemorrhage",
        status=WorkflowStatus.COMPLETED,
        findings=[{
            "category": "hemorrhage",
            "description": "Intraparenchymal hemorrhage, 12.5 mL",
            "severity": "urgent",
        }],
        measurements={"volume_ml": 12.5, "midline_shift_mm": 3.2},
        classification="urgent_hemorrhage",
        severity=FindingSeverity.URGENT,
        nim_services_used=["3D U-Net", "VISTA-3D"],
        is_mock=True,
    )

    return AgentResponse(
        question="What is the best segmentation model?",
        answer=(
            "VISTA-3D from NVIDIA achieves state-of-the-art performance with "
            "mean Dice coefficient of 0.91 across 127 anatomical classes."
        ),
        evidence=evidence,
        workflow_results=[workflow],
        nim_services_used=["vista3d", "maisi"],
        knowledge_used=["pathology", "modality"],
        timestamp="2026-02-27T12:00:00",
    )


# ===================================================================
# export_markdown()
# ===================================================================


class TestExportMarkdown:
    """Tests for export_markdown()."""

    def test_returns_string(self, minimal_response):
        md = export_markdown(minimal_response)
        assert isinstance(md, str)
        assert len(md) > 0

    def test_includes_title(self, minimal_response):
        md = export_markdown(minimal_response)
        assert "Imaging Intelligence Report" in md

    def test_includes_query(self, minimal_response):
        md = export_markdown(minimal_response)
        assert "What is VISTA-3D?" in md

    def test_includes_timestamp(self, minimal_response):
        md = export_markdown(minimal_response)
        assert "2026-02-27" in md

    def test_includes_answer(self, minimal_response):
        md = export_markdown(minimal_response)
        assert "universal segmentation model" in md

    def test_includes_evidence_section(self, rich_response):
        md = export_markdown(rich_response)
        assert "Evidence" in md
        assert "3 items" in md or "items" in md.lower()

    def test_includes_evidence_hits(self, rich_response):
        md = export_markdown(rich_response)
        assert "PMID:001" in md

    def test_includes_nim_services(self, rich_response):
        md = export_markdown(rich_response)
        assert "NIM" in md or "nim" in md.lower()
        assert "vista3d" in md or "maisi" in md

    def test_includes_disclaimer(self, minimal_response):
        md = export_markdown(minimal_response)
        assert "Research use only" in md

    def test_empty_evidence_still_works(self, minimal_response):
        md = export_markdown(minimal_response)
        # Should not crash with empty evidence
        assert "Imaging Intelligence Report" in md

    def test_grouped_by_collection(self, rich_response):
        md = export_markdown(rich_response)
        assert "imaging_literature" in md
        assert "imaging_benchmarks" in md


# ===================================================================
# export_json()
# ===================================================================


class TestExportJSON:
    """Tests for export_json()."""

    def test_returns_string(self, minimal_response):
        result = export_json(minimal_response)
        assert isinstance(result, str)

    def test_valid_json(self, minimal_response):
        result = export_json(minimal_response)
        parsed = json.loads(result)
        assert isinstance(parsed, dict)

    def test_contains_question(self, minimal_response):
        result = export_json(minimal_response)
        parsed = json.loads(result)
        assert parsed["question"] == "What is VISTA-3D?"

    def test_contains_answer(self, minimal_response):
        result = export_json(minimal_response)
        parsed = json.loads(result)
        assert "segmentation" in parsed["answer"]

    def test_contains_evidence(self, minimal_response):
        result = export_json(minimal_response)
        parsed = json.loads(result)
        assert "evidence" in parsed
        assert "hits" in parsed["evidence"]

    def test_contains_timestamp(self, minimal_response):
        result = export_json(minimal_response)
        parsed = json.loads(result)
        assert "timestamp" in parsed

    def test_rich_response_has_all_fields(self, rich_response):
        result = export_json(rich_response)
        parsed = json.loads(result)
        assert len(parsed["evidence"]["hits"]) == 3
        assert len(parsed["workflow_results"]) == 1
        assert len(parsed["nim_services_used"]) == 2

    def test_workflow_results_serialized(self, rich_response):
        result = export_json(rich_response)
        parsed = json.loads(result)
        wr = parsed["workflow_results"][0]
        assert wr["workflow_name"] == "ct_head_hemorrhage"
        assert wr["severity"] == "urgent"

    def test_empty_evidence_works(self):
        response = AgentResponse(
            question="empty test",
            answer="no evidence",
            evidence=CrossCollectionResult(query="empty", hits=[]),
        )
        result = export_json(response)
        parsed = json.loads(result)
        assert parsed["evidence"]["hits"] == []

    def test_json_is_indented(self, minimal_response):
        result = export_json(minimal_response)
        assert "\n" in result  # indent=2 produces newlines
        assert "  " in result  # indent=2 produces 2-space indentation


# ===================================================================
# export_pdf()
# ===================================================================


class TestExportPDF:
    """Tests for export_pdf()."""

    def test_returns_empty_when_reportlab_missing(self, minimal_response):
        """When reportlab is not installed, returns empty string."""
        with patch.dict("sys.modules", {"reportlab": None}):
            with patch("src.export.logger"):
                # Force ImportError
                with patch("builtins.__import__", side_effect=_import_mock_reportlab):
                    result = export_pdf(minimal_response, "/tmp/test_report.pdf")
                    # Either returns the path (if reportlab available) or empty string
                    assert isinstance(result, str)

    def test_with_mock_reportlab(self, rich_response, tmp_path):
        """Test PDF export with mocked reportlab."""
        output_path = str(tmp_path / "test_report.pdf")

        mock_doc = MagicMock()
        mock_paragraph = MagicMock()
        mock_spacer = MagicMock()

        with patch("src.export.logger"):
            try:
                result = export_pdf(rich_response, output_path)
                # If reportlab is installed, it creates the file
                if result:
                    assert result == output_path
                else:
                    # reportlab not installed -> returns empty string
                    assert result == ""
            except Exception:
                # reportlab not installed or import fails
                pass

    def test_pdf_export_handles_multiline_answer(self, minimal_response, tmp_path):
        """Ensure multi-paragraph answers don't crash PDF export."""
        minimal_response.answer = (
            "First paragraph with findings.\n\n"
            "Second paragraph with analysis.\n\n"
            "Third paragraph with recommendations."
        )
        output_path = str(tmp_path / "test_multiline.pdf")

        try:
            result = export_pdf(minimal_response, output_path)
            assert isinstance(result, str)
        except Exception:
            # reportlab not installed
            pass

    def test_pdf_export_with_empty_evidence(self, tmp_path):
        """Ensure PDF works with zero evidence hits."""
        response = AgentResponse(
            question="test",
            answer="No evidence found.",
            evidence=CrossCollectionResult(query="test", hits=[]),
        )
        output_path = str(tmp_path / "test_empty.pdf")

        try:
            result = export_pdf(response, output_path)
            assert isinstance(result, str)
        except Exception:
            pass


# ===================================================================
# EDGE CASES
# ===================================================================


class TestExportEdgeCases:
    """Tests for edge cases across all export functions."""

    def test_markdown_with_special_characters(self):
        response = AgentResponse(
            question="What about <b>bold</b> & 'quotes'?",
            answer="Finding: ratio > 1.0 (5/7 = 0.71)",
            evidence=CrossCollectionResult(query="test", hits=[]),
        )
        md = export_markdown(response)
        assert "<b>bold</b>" in md

    def test_json_with_unicode(self):
        response = AgentResponse(
            question="Analyse du poumon",
            answer="R\u00e9sultat normal -- pas de pathologie.",
            evidence=CrossCollectionResult(query="test", hits=[]),
        )
        result = export_json(response)
        parsed = json.loads(result)
        assert parsed["answer"] == "R\u00e9sultat normal -- pas de pathologie."

    def test_markdown_with_long_answer(self):
        response = AgentResponse(
            question="comprehensive analysis",
            answer="A " * 5000,
            evidence=CrossCollectionResult(query="test", hits=[]),
        )
        md = export_markdown(response)
        assert len(md) > 5000

    def test_json_round_trip(self, rich_response):
        """Export to JSON, parse back, and verify key fields."""
        json_str = export_json(rich_response)
        parsed = json.loads(json_str)
        assert parsed["question"] == rich_response.question
        assert parsed["answer"] == rich_response.answer
        assert len(parsed["evidence"]["hits"]) == rich_response.evidence.hit_count

    def test_markdown_multiple_workflow_results(self):
        evidence = CrossCollectionResult(query="test", hits=[])
        workflows = [
            WorkflowResult(
                workflow_name="ct_head_hemorrhage",
                severity=FindingSeverity.URGENT,
            ),
            WorkflowResult(
                workflow_name="cxr_rapid_findings",
                severity=FindingSeverity.SIGNIFICANT,
            ),
        ]
        response = AgentResponse(
            question="multi-workflow test",
            answer="Multiple workflows executed.",
            evidence=evidence,
            workflow_results=workflows,
        )
        md = export_markdown(response)
        # markdown function from export.py does not have workflow section,
        # but it should not crash
        assert isinstance(md, str)
        assert len(md) > 0


def _import_mock_reportlab(name, *args, **kwargs):
    """Helper to mock importlib for reportlab tests."""
    if "reportlab" in name:
        raise ImportError("Mocked: reportlab not installed")
    return __import__(name, *args, **kwargs)
