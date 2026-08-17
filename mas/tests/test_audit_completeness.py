"""Audit-log completeness metric — CLAUDE.md's audit convention, measured.

The convention says EVERY node must append a LogEntry to `state["audit"]`. A node
that runs silently leaves a hole in the trail a human auditor would follow, so the
KPI is not "how many entries were written" (a chatty node would mask a silent one)
but **"did every node that ran actually log?"**

RunMetrics therefore tracks two sets — nodes that ran, and nodes that logged — and
`audit_complete` is true only when the first is a subset of the second. The raw
nodes_audited / nodes_total ride along so the report can aggregate a rate across
rows rather than only a boolean.
"""

import pytest

from utils.telemetry.metrics import FIELDNAMES, RunMetrics


def test_all_nodes_logging_is_complete():
    m = RunMetrics(source="test")
    # All three are nodes WE author and are bound by the audit convention. The
    # built-in ToolNode ("tools") is deliberately NOT used here — it is audit-
    # exempt (its work is logged by the adjacent *_llm node), so saw_node drops
    # it from the denominator; see _AUDIT_EXEMPT_NODES and test_tools_node_is_exempt.
    for node in ("orchestrator", "borrower_profile_llm", "compliance_validation_llm"):
        m.saw_node(node)
        m.add_audit_entries(node, 3)

    assert m.nodes_total == 3
    assert m.nodes_audited == 3
    assert m.audit_complete is True
    assert m.audit_entries == 9
    assert m.unaudited_nodes == []


def test_tools_node_is_exempt():
    """The built-in ToolNode ("tools") is excluded from the completeness denominator.

    It structurally cannot append to state["audit"] (it returns only messages), and
    its tool call is already audited by the adjacent *_llm node. Counting it as an
    unaudited node would pin audit_complete below 100% on every turn that uses a tool
    — measuring a framework fact, not a gap in our trail. See _AUDIT_EXEMPT_NODES.
    """
    m = RunMetrics(source="test")
    m.saw_node("orchestrator")
    m.add_audit_entries("orchestrator", 3)
    m.saw_node("tools")            # ran, but is audit-exempt
    m.add_audit_entries("tools", 0)

    assert m.nodes_total == 1      # only the orchestrator counts
    assert m.audit_complete is True
    assert "tools" not in m.unaudited_nodes


def test_a_silent_node_breaks_completeness():
    """The actual defect: a node runs but writes nothing to the audit trail."""
    m = RunMetrics(source="test")
    m.saw_node("orchestrator")
    m.add_audit_entries("orchestrator", 3)
    m.saw_node("property_analysis_llm")
    m.add_audit_entries("property_analysis_llm", 0)   # ran, logged nothing

    assert m.nodes_total == 2
    assert m.nodes_audited == 1
    assert m.audit_complete is False
    assert m.unaudited_nodes == ["property_analysis_llm"]


def test_entry_count_alone_cannot_hide_a_silent_node():
    """A chatty node must not mask a silent one — which is why we count NODES.

    Here the audit trail has 20 entries, which looks healthy, but one node
    contributed none of them. A naive "audit_entries > 0" check would pass.
    """
    m = RunMetrics(source="test")
    m.saw_node("orchestrator")
    m.add_audit_entries("orchestrator", 20)
    m.saw_node("compliance_validation_llm")
    m.add_audit_entries("compliance_validation_llm", 0)

    assert m.audit_entries == 20          # looks fine
    assert m.audit_complete is False      # but it is not


def test_repeated_node_visits_count_once():
    """A tool loop re-enters the same llm node; it is still one node, not two."""
    m = RunMetrics(source="test")
    m.saw_node("borrower_profile_llm")
    m.add_audit_entries("borrower_profile_llm", 2)
    m.saw_node("borrower_profile_llm")          # loop re-entry
    m.add_audit_entries("borrower_profile_llm", 3)

    assert m.nodes_total == 1
    assert m.audit_entries == 5
    assert m.audit_complete is True


def test_empty_stream_is_vacuously_complete_but_reports_zero_nodes():
    """An errored stream ran no nodes. audit_complete is vacuously true, so the
    report must aggregate over rows with nodes_total > 0 — hence writing the raw
    counts, not just the boolean."""
    m = RunMetrics(source="test")
    assert m.audit_complete is True
    assert m.nodes_total == 0
    assert m.nodes_audited == 0


def test_row_carries_the_audit_columns():
    m = RunMetrics(source="test")
    m.saw_node("orchestrator")
    m.add_audit_entries("orchestrator", 3)
    m.saw_node("hitl_review")
    m.add_audit_entries("hitl_review", 0)

    row = m.to_row()
    for col in ("audit_entries", "nodes_audited", "nodes_total", "audit_complete"):
        assert col in FIELDNAMES, f"{col} missing from the runs.csv schema"
        assert col in row

    assert row["audit_entries"] == 3
    assert row["nodes_total"] == 2
    assert row["nodes_audited"] == 1
    assert row["audit_complete"] == 0     # written as 0/1, not True/False


def test_row_key_order_matches_the_schema():
    """to_row() must emit exactly FIELDNAMES — a mismatch silently shifts columns
    in runs.csv (DictWriter would raise, but only after the header was written)."""
    m = RunMetrics(source="test")
    assert set(m.to_row()) == set(FIELDNAMES)
