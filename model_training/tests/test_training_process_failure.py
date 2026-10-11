from __future__ import annotations

import subprocess
import sys

import pytest

from naartjie_cv.train import TrainingProcessError, _run_training_process


@pytest.mark.parametrize("use_log", [True, False])
def test_failed_process_surfaces_ram_error_and_recovery_instructions(tmp_path, use_log):
    log = tmp_path / "trainer_console.log" if use_log else None
    with pytest.raises(subprocess.CalledProcessError) as caught:
        _run_training_process(
            [sys.executable, "-c", "raise MemoryError('Unable to allocate 1.56 MiB')"],
            cwd=tmp_path, output_callback=lambda _: None, output_log=log,
        )
    error = caught.value
    assert error.returncode == 1
    assert "MemoryError: Unable to allocate 1.56 MiB" in str(error)
    assert "Data workers to 0" in str(error)
    assert "--workers 0" in str(error)
    assert "returned non-zero exit status" not in str(error)
    if use_log:
        assert str(log) in str(error)
        assert "MemoryError" in log.read_text()


def test_failed_process_ignores_errors_from_an_earlier_log_session(tmp_path):
    log = tmp_path / "trainer_console.log"
    log.write_text("Traceback (most recent call last):\nMemoryError: Unable to allocate old array\n")
    with pytest.raises(TrainingProcessError) as caught:
        _run_training_process(
            [sys.executable, "-c", "raise ValueError('invalid class mapping')"],
            cwd=tmp_path, output_callback=lambda _: None, output_log=log,
        )
    message = str(caught.value)
    assert "ValueError: invalid class mapping" in message
    assert "old array" not in message
    assert "System RAM" not in message


def test_failed_process_still_reports_error_when_dashboard_callback_fails(tmp_path):
    def disconnected(_):
        raise RuntimeError("dashboard disconnected")

    with pytest.raises(TrainingProcessError, match="ValueError: invalid annotations"):
        _run_training_process(
            [sys.executable, "-c", "raise ValueError('invalid annotations')"],
            cwd=tmp_path, output_callback=disconnected, output_log=tmp_path / "console.log",
        )


def test_error_message_keeps_cuda_failure_separate_from_system_ram():
    error = TrainingProcessError(1, ["python"], "torch.OutOfMemoryError: CUDA out of memory")
    assert "CUDA out of memory" in str(error)
    assert "Data workers" not in str(error)


@pytest.mark.parametrize("use_log", [True, False])
def test_failed_worker_surfaces_recovery_without_claiming_ram_exhaustion(tmp_path, use_log):
    log = tmp_path / "trainer_console.log" if use_log else None
    with pytest.raises(TrainingProcessError) as caught:
        _run_training_process(
            [sys.executable, "-c", "raise RuntimeError('DataLoader worker (pid(s) 16780) exited unexpectedly')"],
            cwd=tmp_path, output_callback=lambda _: None, output_log=log,
        )
    message = str(caught.value)
    assert "DataLoader worker (pid(s) 16780) exited unexpectedly" in message
    assert "Data workers to 0" in message and "--workers 0" in message
    assert "Restart the notebook kernel" in message
    assert "System RAM allocation failed" not in message
