"""Export agent responses to Markdown, JSON, and PDF formats."""

import json
from typing import Optional

from loguru import logger

from src.models import AgentResponse


def export_markdown(response: AgentResponse) -> str:
    """Export response as Markdown string."""
    # (Similar to agent.generate_report but more detailed)
    md = [
        f"# Imaging Intelligence Report\n",
        f"**Query:** {response.question}\n",
        f"**Timestamp:** {response.timestamp}\n",
        f"\n## Analysis\n\n{response.answer}\n",
        f"\n## Evidence ({response.evidence.hit_count} items)\n",
    ]

    for collection, hits in response.evidence.hits_by_collection().items():
        md.append(f"\n### {collection} ({len(hits)} results)\n")
        for hit in hits[:5]:
            md.append(f"- [{hit.id}] (score: {hit.score:.3f}) {hit.text[:200]}...\n")

    if response.nim_services_used:
        md.append(f"\n## NVIDIA NIM Services Used\n")
        for nim in response.nim_services_used:
            md.append(f"- {nim}\n")

    md.append(f"\n---\n*Research use only.*\n")
    return "".join(md)


def export_json(response: AgentResponse) -> str:
    """Export response as JSON string."""
    return response.model_dump_json(indent=2)


def export_pdf(response: AgentResponse, output_path: str) -> str:
    """Export response as PDF file."""
    try:
        from reportlab.lib.pagesizes import letter
        from reportlab.lib.styles import getSampleStyleSheet
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer

        doc = SimpleDocTemplate(output_path, pagesize=letter)
        styles = getSampleStyleSheet()
        story = []

        story.append(Paragraph("Imaging Intelligence Report", styles["Title"]))
        story.append(Spacer(1, 12))
        story.append(Paragraph(f"<b>Query:</b> {response.question}", styles["Normal"]))
        story.append(Paragraph(f"<b>Timestamp:</b> {response.timestamp}", styles["Normal"]))
        story.append(Spacer(1, 12))
        story.append(Paragraph("Analysis", styles["Heading2"]))

        # Split answer into paragraphs
        for para in response.answer.split("\n\n"):
            if para.strip():
                story.append(Paragraph(para.strip(), styles["Normal"]))
                story.append(Spacer(1, 6))

        story.append(Spacer(1, 12))
        story.append(Paragraph(f"Evidence: {response.evidence.hit_count} items from {response.evidence.total_collections_searched} collections", styles["Normal"]))
        story.append(Spacer(1, 12))
        story.append(Paragraph("<i>Research use only. All findings require clinician review.</i>", styles["Normal"]))

        doc.build(story)
        logger.info(f"PDF exported to {output_path}")
        return output_path
    except ImportError:
        logger.error("reportlab not installed - cannot export PDF")
        return ""
