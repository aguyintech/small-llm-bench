"""Tests for the mock tool registry and executor."""

from __future__ import annotations

import pytest

from small_llm_bench.modules.mock_registry import (TOOL_IMPLS, TOOL_SCHEMAS,
                                                   ToolExecutor)

SAMPLE_CALLS = {
    "get_weather": {"city": "Tokyo"},
    "get_calendar_events": {"date": "2026-06-08"},
    "create_event": {"title": "T", "date": "2026-06-08", "time": "10:00"},
    "search_web": {"query": "python"},
    "read_file": {"path": "/data/report.txt"},
    "write_file": {"path": "/tmp/x.txt", "content": "abc"},
    "get_stock_price": {"symbol": "AAPL"},
    "send_message": {"recipient": "a@example.com", "message": "hi"},
    "calculate": {"expression": "2 + 3 * 4"},
    "get_contacts": {"name": "Alice"},
    "db_query": {"table": "orders", "filters": {"status": "shipped"}},
    "create_ticket": {"title": "T", "fields": {"priority": "high"}},
    "run_deploy": {"service": "checkout", "env": "prod", "replicas": 3},
    "post_update": {"channel": "#ops", "content": "hi"},
}

EXPECTED_KEYS = {
    "get_weather": {"city", "temperature", "condition", "humidity"},
    "create_event": {"event_id", "status", "title"},
    "read_file": {"path", "content"},
    "write_file": {"status", "path", "bytes_written"},
    "get_stock_price": {"symbol", "price", "currency"},
    "send_message": {"status", "message_id"},
    "calculate": {"expression", "result"},
    "db_query": {"table", "count", "rows"},
    "create_ticket": {"ticket_id", "status", "title", "fields"},
    "run_deploy": {"service", "env", "replicas", "status"},
    "post_update": {"status", "channel", "chars"},
}


def test_every_impl_has_schema_and_vice_versa():
    assert set(TOOL_IMPLS) == set(TOOL_SCHEMAS)


@pytest.mark.parametrize("name", sorted(SAMPLE_CALLS))
def test_tool_returns_expected_structure(name):
    result = TOOL_IMPLS[name](**SAMPLE_CALLS[name])
    if name in EXPECTED_KEYS:
        assert EXPECTED_KEYS[name] <= set(result)
    else:
        assert isinstance(result, list)
        assert all(isinstance(item, dict) for item in result)


def test_tools_are_deterministic():
    first = TOOL_IMPLS["get_weather"](city="Tokyo")
    second = TOOL_IMPLS["get_weather"](city="Tokyo")
    assert first == second


def test_calculate_result():
    assert TOOL_IMPLS["calculate"](expression="2 + 3 * 4")["result"] == 14


def test_calculate_rejects_non_arithmetic():
    assert "error" in TOOL_IMPLS["calculate"](expression="__import__('os')")


def test_get_contacts_filters_by_name():
    contacts = TOOL_IMPLS["get_contacts"](name="dana")
    assert len(contacts) == 1
    assert contacts[0]["email"] == "dana@example.com"


def test_executor_first_call_error_then_success():
    executor = ToolExecutor({"read_file": {"first_call_behavior": "error"}})
    first = executor.execute("read_file", {"path": "/data/report.txt"})
    second = executor.execute("read_file", {"path": "/data/report.txt"})
    assert "error" in first
    assert "content" in second


def test_executor_first_call_empty():
    executor = ToolExecutor({"get_contacts": {"first_call_behavior": "empty"}})
    assert executor.execute("get_contacts", {}) == []


def test_executor_unknown_tool():
    executor = ToolExecutor()
    assert "error" in executor.execute("nope", {})


def test_executor_invalid_arguments():
    executor = ToolExecutor()
    assert "error" in executor.execute("get_weather", {"bogus": 1})
