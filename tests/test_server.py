"""Web service hygiene: inputs that name files, and state that only grows.

``image_id`` was joined onto the uploads folder unchecked, so an absolute path
or ``..`` rendered any image on the machine. The job table and the uploads,
outputs and graph-cache folders were never trimmed, so a long-running service
only grew.
"""

from __future__ import annotations

import os
import time

import pytest
from fastapi.testclient import TestClient

from webapp import server


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(server, "UPLOADS", tmp_path)
    monkeypatch.setattr(server, "_run_job", lambda job_id, req: None)
    monkeypatch.setattr(server, "_jobs", type(server._jobs)())
    return TestClient(server.app)


@pytest.mark.parametrize("image_id", ["/etc/hosts", "../server.py", "a/b.png", "abc.exe"])
def test_render_rejects_an_image_id_that_is_not_an_upload(client, image_id):
    r = client.post("/api/render", json={"image_id": image_id})
    assert r.status_code == 422


def _render(client):
    return client.post("/api/render", json={"image_id": "0123456789ab.png"}).json()["job_id"]


def test_job_table_drops_the_oldest_finished_jobs(client, tmp_path, monkeypatch):
    monkeypatch.setattr(server, "MAX_JOBS", 3)
    (tmp_path / "0123456789ab.png").write_bytes(b"x")

    ids = []
    for _ in range(5):
        ids.append(_render(client))
        server._jobs[ids[-1]]["state"] = "done"

    assert list(server._jobs) == ids[-3:]


def test_job_table_never_drops_a_job_still_waiting_to_run(client, tmp_path, monkeypatch):
    """A dropped queued job dies silently and its poller gets a 404."""
    monkeypatch.setattr(server, "MAX_JOBS", 2)
    (tmp_path / "0123456789ab.png").write_bytes(b"x")

    ids = [_render(client) for _ in range(4)]

    assert list(server._jobs) == ids


def test_an_upload_in_use_is_not_pruned(client, tmp_path):
    """Age is measured from last use, so an open tab's photo survives."""
    upload = tmp_path / "0123456789ab.png"
    upload.write_bytes(b"x")
    stale = time.time() - 10 * 86400
    os.utime(upload, (stale, stale))

    _render(client)
    server._prune_old_files(tmp_path, max_age_s=86400)

    assert upload.exists()


def test_prune_removes_old_files_and_keeps_recent_ones(tmp_path):
    cache = tmp_path / ".heron_graph"
    cache.mkdir()
    old, fresh, old_graph = tmp_path / "old.png", tmp_path / "fresh.png", cache / "old.npz"
    for f in (old, fresh, old_graph):
        f.write_bytes(b"x")
    stale = time.time() - 10 * 86400
    for f in (old, old_graph):
        os.utime(f, (stale, stale))

    server._prune_old_files(tmp_path, max_age_s=86400)

    assert not old.exists() and not old_graph.exists()
    assert fresh.exists()
