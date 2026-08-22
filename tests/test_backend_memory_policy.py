from __future__ import annotations

from pathlib import Path

from ltx_worker import _configure_main_pipeline
from wan_worker import _configure_memory


class _WanPipe:
    def __init__(self):
        self.calls = []

    def enable_group_offload(self, **kwargs):
        self.calls.append(("group", kwargs))

    def enable_sequential_cpu_offload(self, **kwargs):
        self.calls.append(("sequential", kwargs))

    def enable_model_cpu_offload(self, **kwargs):
        self.calls.append(("model", kwargs))

    def to(self, device):
        self.calls.append(("to", device))


class _WanPipeWithoutGroup:
    def __init__(self):
        self.calls = []

    def enable_sequential_cpu_offload(self, **kwargs):
        self.calls.append(("sequential", kwargs))

    def enable_model_cpu_offload(self, **kwargs):
        self.calls.append(("model", kwargs))

    def to(self, device):
        self.calls.append(("to", device))


class _LTXPipe:
    def __init__(self):
        self.calls = []

    def enable_sequential_cpu_offload(self, **kwargs):
        self.calls.append(("sequential", kwargs))

    def enable_model_cpu_offload(self, **kwargs):
        self.calls.append(("model", kwargs))

    def to(self, device):
        self.calls.append(("to", device))


def test_wan_auto_prefers_group_offload_on_16gb_gpu(monkeypatch, tmp_path):
    monkeypatch.setenv("DUCKMOTION_WAN_OFFLOAD", "auto")
    monkeypatch.setattr("wan_worker._source_size_gb", lambda _path: 34.0)
    monkeypatch.setattr("wan_worker._system_memory_gb", lambda: 64.0)
    pipe = _WanPipe()

    mode = _configure_memory(pipe, "cuda", 16.0, tmp_path, "/models/wan")

    assert mode == "group"
    assert pipe.calls[0][0] == "group"
    assert pipe.calls[0][1]["offload_type"] == "leaf_level"


def test_wan_group_policy_falls_back_to_sequential_if_group_api_missing(monkeypatch, tmp_path):
    monkeypatch.setenv("DUCKMOTION_WAN_OFFLOAD", "auto")
    pipe = _WanPipeWithoutGroup()

    mode = _configure_memory(pipe, "cuda", 16.0, tmp_path, "/models/wan")

    assert mode == "sequential"
    assert pipe.calls == [("sequential", {"device": "cuda"})]


def test_wan_large_local_source_can_request_disk_backed_group_offload(monkeypatch, tmp_path):
    monkeypatch.setenv("DUCKMOTION_WAN_OFFLOAD", "group")
    monkeypatch.setattr("wan_worker._source_size_gb", lambda _path: 126.0)
    monkeypatch.setattr("wan_worker._system_memory_gb", lambda: 64.0)
    pipe = _WanPipe()

    mode = _configure_memory(pipe, "cuda", 16.0, tmp_path, "/models/wan-a14b")

    assert mode == "group"
    kwargs = pipe.calls[0][1]
    assert Path(kwargs["offload_to_disk_path"]).name == ".wan_offload"


def test_ltx_auto_uses_sequential_offload_on_16gb_gpu(monkeypatch):
    monkeypatch.setenv("DUCKMOTION_LTX_OFFLOAD", "auto")
    pipe = _LTXPipe()

    mode = _configure_main_pipeline(pipe, "cuda", 16.0)

    assert mode == "sequential"
    assert pipe.calls == [("sequential", {"device": "cuda"})]


def test_ltx_auto_can_use_model_offload_on_large_gpu(monkeypatch):
    monkeypatch.setenv("DUCKMOTION_LTX_OFFLOAD", "auto")
    pipe = _LTXPipe()

    mode = _configure_main_pipeline(pipe, "cuda", 48.0)

    assert mode == "model"
    assert pipe.calls == [("model", {"device": "cuda"})]
