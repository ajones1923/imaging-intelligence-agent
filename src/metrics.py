"""Prometheus metrics for Imaging Intelligence Agent monitoring.

Exposes counters, histograms, and gauges for query performance,
NIM service usage, and collection health tracking.
"""

from typing import Dict, Optional

from loguru import logger

try:
    from prometheus_client import Counter, Gauge, Histogram
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    logger.warning("prometheus_client not installed — metrics will be no-ops")


# ═══════════════════════════════════════════════════════════════════════
# METRIC DEFINITIONS
# ═══════════════════════════════════════════════════════════════════════

if PROMETHEUS_AVAILABLE:
    QUERY_COUNT = Counter(
        "imaging_agent_query_total",
        "Total number of RAG queries processed",
        ["query_type", "modality"],
    )

    QUERY_LATENCY = Histogram(
        "imaging_agent_query_latency_seconds",
        "RAG query end-to-end latency in seconds",
        ["query_type"],
        buckets=(0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0),
    )

    COLLECTION_SIZE = Gauge(
        "imaging_agent_collection_size",
        "Number of vectors in each Milvus collection",
        ["collection"],
    )

    NIM_REQUEST_COUNT = Counter(
        "imaging_agent_nim_request_total",
        "Total number of NIM service invocations",
        ["nim_service", "status"],
    )

    NIM_REQUEST_LATENCY = Histogram(
        "imaging_agent_nim_request_latency_seconds",
        "NIM service invocation latency in seconds",
        ["nim_service"],
        buckets=(0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0),
    )
else:
    # No-op fallbacks when prometheus_client is not installed
    class _NoOpMetric:
        def labels(self, *args, **kwargs):
            return self
        def inc(self, *args, **kwargs):
            pass
        def observe(self, *args, **kwargs):
            pass
        def set(self, *args, **kwargs):
            pass

    QUERY_COUNT = _NoOpMetric()
    QUERY_LATENCY = _NoOpMetric()
    COLLECTION_SIZE = _NoOpMetric()
    NIM_REQUEST_COUNT = _NoOpMetric()
    NIM_REQUEST_LATENCY = _NoOpMetric()


# ═══════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════


def record_query(query_type: str = "standard", modality: str = "any", latency_seconds: float = 0.0) -> None:
    """Record a completed RAG query with its type, modality, and latency.

    Args:
        query_type: One of "standard", "comparative", "targeted", "streaming".
        modality: The imaging modality filter used (e.g., "ct", "mri", "any").
        latency_seconds: End-to-end query latency in seconds.
    """
    QUERY_COUNT.labels(query_type=query_type, modality=modality).inc()
    QUERY_LATENCY.labels(query_type=query_type).observe(latency_seconds)


def record_nim_request(nim_service: str, status: str = "success", latency_seconds: float = 0.0) -> None:
    """Record a NIM service invocation.

    Args:
        nim_service: Name of the NIM service (e.g., "vista3d", "maisi", "vilam3").
        status: One of "success", "error", "timeout", "mock".
        latency_seconds: NIM invocation latency in seconds.
    """
    NIM_REQUEST_COUNT.labels(nim_service=nim_service, status=status).inc()
    NIM_REQUEST_LATENCY.labels(nim_service=nim_service).observe(latency_seconds)


def update_collection_sizes(collection_manager, collections: Optional[Dict[str, str]] = None) -> Dict[str, int]:
    """Update Prometheus gauges with current Milvus collection sizes.

    Args:
        collection_manager: The Milvus collection manager instance.
        collections: Optional dict of {collection_name: label}. If None, uses
            default imaging collections.

    Returns:
        Dict mapping collection name to entity count.
    """
    if collections is None:
        collections = {
            "imaging_literature": "Literature",
            "imaging_trials": "Trials",
            "imaging_findings": "Findings",
            "imaging_protocols": "Protocols",
            "imaging_devices": "Devices",
            "imaging_anatomy": "Anatomy",
            "imaging_benchmarks": "Benchmarks",
            "imaging_guidelines": "Guidelines",
            "imaging_report_templates": "ReportTemplates",
            "imaging_datasets": "Datasets",
            "genomic_evidence": "Genomic",
        }

    sizes = {}
    for coll_name in collections:
        try:
            count = collection_manager.get_entity_count(coll_name)
            COLLECTION_SIZE.labels(collection=coll_name).set(count)
            sizes[coll_name] = count
        except Exception as e:
            logger.warning(f"Failed to get size for {coll_name}: {e}")
            sizes[coll_name] = -1

    return sizes
