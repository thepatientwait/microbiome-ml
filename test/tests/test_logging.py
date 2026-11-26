"""Tests for the logging helpers.

``tee_log`` redirects file descriptors 1 and 2, which pytest's own output
capture interferes with, so it is exercised in a child interpreter.
"""

import subprocess
import sys
import textwrap


def _run(code: str):
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)],
        capture_output=True,
        text=True,
    )


def test_import_does_not_configure_root_logging():
    out = _run(
        """
        import logging, microbiome_ml
        print(len(logging.getLogger().handlers))
        """
    )
    assert out.stdout.strip() == "0", out.stderr


def test_tee_log_captures_print_subprocess_and_stderr(tmp_path):
    log = tmp_path / "logs" / "job.log"
    out = _run(
        f"""
        import subprocess, sys
        from microbiome_ml.utils.logging import tee_log
        with tee_log({str(log)!r}, label="rule_x"):
            print("hello from python")
            subprocess.run([sys.executable, "-c", "print('hello from child')"])
            sys.stderr.write("to stderr\\n")
        print("after")
        """
    )
    text = log.read_text()
    assert "rule_x, attempt 1" in text
    assert "hello from python" in text
    assert "hello from child" in text
    assert "to stderr" in text
    # descriptors restored: later output goes to the real stdout, not the log
    assert "after" not in text
    assert out.stdout.strip() == "after"


def test_tee_log_attempt_one_truncates_later_attempts_append(tmp_path):
    log = tmp_path / "job.log"
    _run(
        f"""
        from microbiome_ml.utils.logging import tee_log
        with tee_log({str(log)!r}, attempt=1):
            print("first")
        with tee_log({str(log)!r}, attempt=2):
            print("second")
        """
    )
    text = log.read_text()
    assert "first" in text and "second" in text and "attempt 2" in text
    _run(
        f"""
        from microbiome_ml.utils.logging import tee_log
        with tee_log({str(log)!r}, attempt=1):
            print("fresh")
        """
    )
    text = log.read_text()
    assert "fresh" in text and "first" not in text


def test_tee_log_writes_traceback_and_still_raises(tmp_path):
    log = tmp_path / "job.log"
    out = _run(
        f"""
        from microbiome_ml.utils.logging import tee_log
        try:
            with tee_log({str(log)!r}):
                raise ValueError("boom")
        except ValueError:
            print("caught")
        """
    )
    assert "ValueError: boom" in log.read_text()
    assert out.stdout.strip() == "caught"
