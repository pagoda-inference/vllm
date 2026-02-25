# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for Pagoda request logger and request timer."""

import json
import logging
import time

import pytest

from vllm.pagoda.request_logger import (
    PagodaRequestLog,
    PagodaRequestLogger,
    RequestTimer,
)


class TestPagodaRequestLog:
    """Tests for the PagodaRequestLog dataclass."""

    def test_defaults(self):
        """Test default field values."""
        log = PagodaRequestLog(
            request_id="req-1",
            tenant_id="t-1",
            model="llama-3",
            priority="normal",
        )
        assert log.status == "success"
        assert log.prompt_tokens == 0
        assert log.output_tokens == 0
        assert log.tool_call is False
        assert log.preempted is False
        assert log.error_reason is None
        assert log.timestamp  # auto-populated

    def test_timestamp_auto_populated(self):
        """Test that timestamp is set automatically if not provided."""
        log = PagodaRequestLog(
            request_id="req-1",
            tenant_id="t-1",
            model="m",
            priority="normal",
        )
        assert log.timestamp.endswith("+00:00") or log.timestamp.endswith("Z")

    def test_custom_timestamp_preserved(self):
        """Test that an explicit timestamp is not overwritten."""
        ts = "2025-01-01T00:00:00.000+00:00"
        log = PagodaRequestLog(
            request_id="req-1",
            tenant_id="t-1",
            model="m",
            priority="normal",
            timestamp=ts,
        )
        assert log.timestamp == ts

    def test_user_id_default_none(self):
        """Test that user_id defaults to None."""
        log = PagodaRequestLog(
            request_id="req-1",
            tenant_id="t-1",
            model="m",
            priority="normal",
        )
        assert log.user_id is None

    def test_user_id_set(self):
        """Test that user_id can be set explicitly."""
        log = PagodaRequestLog(
            request_id="req-1",
            tenant_id="t-1",
            model="m",
            priority="normal",
            user_id="user-42",
        )
        assert log.user_id == "user-42"


class TestPagodaRequestLogger:
    """Tests for the PagodaRequestLogger."""

    def test_emits_json_line(self, caplog):
        """Test that log() emits a valid JSON line."""
        logger_inst = PagodaRequestLogger(logger_name="test.request_log_1")
        entry = PagodaRequestLog(
            request_id="req-42",
            tenant_id="tenant-abc",
            model="llama-3",
            priority="high",
            prompt_tokens=100,
            output_tokens=50,
            total_ms=123.4,
            status="success",
        )

        with caplog.at_level(logging.INFO, logger="test.request_log_1"):
            logger_inst.log(entry)

        assert len(caplog.records) == 1
        data = json.loads(caplog.records[0].message)
        assert data["request_id"] == "req-42"
        assert data["tenant_id"] == "tenant-abc"
        assert data["model"] == "llama-3"
        assert data["prompt_tokens"] == 100
        assert data["total_ms"] == 123.4

    def test_none_fields_omitted(self, caplog):
        """Test that None-valued fields are dropped from JSON output."""
        logger_inst = PagodaRequestLogger(logger_name="test.request_log_2")
        entry = PagodaRequestLog(
            request_id="req-1",
            tenant_id="t-1",
            model="m",
            priority="normal",
            error_reason=None,
        )

        with caplog.at_level(logging.INFO, logger="test.request_log_2"):
            logger_inst.log(entry)

        data = json.loads(caplog.records[0].message)
        assert "error_reason" not in data

    def test_error_reason_included_when_set(self, caplog):
        """Test that error_reason appears when it has a value."""
        logger_inst = PagodaRequestLogger(logger_name="test.request_log_3")
        entry = PagodaRequestLog(
            request_id="req-1",
            tenant_id="t-1",
            model="m",
            priority="normal",
            status="rejected",
            error_reason="rate_limit:qps_limit",
        )

        with caplog.at_level(logging.INFO, logger="test.request_log_3"):
            logger_inst.log(entry)

        data = json.loads(caplog.records[0].message)
        assert data["error_reason"] == "rate_limit:qps_limit"

    def test_user_id_included_in_json(self, caplog):
        """Test that user_id appears in JSON output when set."""
        logger_inst = PagodaRequestLogger(logger_name="test.request_log_4")
        entry = PagodaRequestLog(
            request_id="req-1",
            tenant_id="t-1",
            model="m",
            priority="normal",
            user_id="user-42",
        )

        with caplog.at_level(logging.INFO, logger="test.request_log_4"):
            logger_inst.log(entry)

        data = json.loads(caplog.records[0].message)
        assert data["user_id"] == "user-42"

    def test_user_id_omitted_when_none(self, caplog):
        """Test that user_id is omitted from JSON when None."""
        logger_inst = PagodaRequestLogger(logger_name="test.request_log_5")
        entry = PagodaRequestLog(
            request_id="req-1",
            tenant_id="t-1",
            model="m",
            priority="normal",
        )

        with caplog.at_level(logging.INFO, logger="test.request_log_5"):
            logger_inst.log(entry)

        data = json.loads(caplog.records[0].message)
        assert "user_id" not in data


class TestRequestTimer:
    """Tests for the RequestTimer helper."""

    def test_total_ms_increases(self):
        """Test that total_ms reflects elapsed time."""
        timer = RequestTimer()
        time.sleep(0.01)
        timer.mark_done()
        assert timer.total_ms >= 10.0

    def test_queue_wait_and_inference(self):
        """Test that queue_wait_ms and inference_ms are tracked."""
        timer = RequestTimer()
        time.sleep(0.01)
        timer.mark_inference_start()
        time.sleep(0.01)
        timer.mark_done()

        assert timer.queue_wait_ms >= 10.0
        assert timer.inference_ms >= 10.0
        assert timer.total_ms >= 20.0

    def test_no_inference_start_gives_zero_queue_wait(self):
        """Test that queue_wait_ms is 0 if inference never started."""
        timer = RequestTimer()
        timer.mark_done()
        assert timer.queue_wait_ms == 0.0

    def test_no_done_gives_zero_inference(self):
        """Test that inference_ms is 0 if mark_done was not called."""
        timer = RequestTimer()
        timer.mark_inference_start()
        assert timer.inference_ms == 0.0

    def test_total_ms_live_before_done(self):
        """Test that total_ms returns live value before mark_done."""
        timer = RequestTimer()
        time.sleep(0.01)
        assert timer.total_ms >= 10.0
