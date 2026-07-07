from __future__ import annotations

from scripts.run_dense import compare


def test_compare_reports_deltas_for_shared_metrics():
    baseline = {"recall_at_k": 0.5, "precision_at_k": 0.2, "mrr": 0.4}
    dense = {"recall_at_k": 0.8, "precision_at_k": 0.5, "mrr": 0.7}
    rows = compare(baseline, dense)
    by_metric = {r["metric"]: r for r in rows}
    assert abs(by_metric["recall_at_k"]["delta"] - 0.3) < 1e-9
    assert by_metric["recall_at_k"]["baseline"] == 0.5
    assert by_metric["recall_at_k"]["dense"] == 0.8
