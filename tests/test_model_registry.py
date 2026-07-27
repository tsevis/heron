"""Model resolution on a machine that is not the one this was written on.

The engine used to name absolute paths under `~/AI/ClaudeCode/mozaix/`. On a
fresh clone every lookup failed, the AI path silently never engaged, and the
classical fallback took over without saying so — a worse render and no
explanation. These tests pin down the resolution order and, more importantly,
that a machine with nothing installed is *reported* rather than silently
degraded.

They simulate a bare machine by pointing every lookup at empty directories,
because the development machine has all of this and would otherwise pass
vacuously.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from heron.scene.ai import registry

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture
def bare(monkeypatch, tmp_path):
    """A machine with no models anywhere: no override, no repo dir, no cache."""
    monkeypatch.setenv("HERON_MODELS", str(tmp_path / "nothing"))
    monkeypatch.setenv("HF_HOME", str(tmp_path / "no-cache"))
    monkeypatch.setattr(registry, "PROJECT_MODELS", tmp_path / "no-models")
    # Legacy paths are absolute and machine-specific; blank them for this test.
    monkeypatch.setattr(
        registry, "SPECS",
        tuple(type(s)(**{**s.__dict__, "legacy": ()}) for s in registry.SPECS))
    monkeypatch.setattr(registry, "BY_ROLE", {s.role: s for s in registry.SPECS})
    return tmp_path


def _hf_dir(path: Path) -> Path:
    """A directory that looks like a downloaded HF model."""
    path.mkdir(parents=True, exist_ok=True)
    (path / "config.json").write_text("{}")
    return path


def test_bare_machine_reports_everything_missing(bare):
    """The failure this whole module exists to prevent: silent degradation."""
    for role in ("depth", "matte", "segment"):
        assert registry.resolve(role) is None, role
    assert {s.role for s in registry.missing()} == {"depth", "matte", "segment"}


def test_marker_file_decides_whether_a_model_counts(bare, monkeypatch):
    """SAM 3 ships config.json but is unusable without sam3.pt.

    Checking for config.json alone would call a partial download "installed",
    and the failure would surface much later as an obscure loader error.
    """
    models = bare / "no-models"
    sam3 = models / registry.BY_ROLE["segment"].dir_name
    _hf_dir(sam3)                      # config.json only — not enough
    monkeypatch.setattr(registry, "PROJECT_MODELS", models)
    assert registry.resolve("segment") is None

    (sam3 / "sam3.pt").write_bytes(b"")
    assert registry.resolve("segment") == str(sam3)


def test_gated_models_carry_an_explanation():
    """A 403 with no guidance is a dead end for anyone cloning this."""
    for spec in registry.SPECS:
        if spec.gated:
            assert spec.note, f"{spec.role} is gated but explains nothing"


def test_sam3_fetches_its_python_package_not_just_weights():
    """SAM 3 is imported by path, so a weights-only allow-list would break it."""
    spec = registry.BY_ROLE["segment"]
    assert spec.patterns == (), "an allow-list would drop SAM 3's own package"
    assert "model.safetensors" in spec.ignore, (
        "the repo ships two INCOMPATIBLE serializations of the same ~860M-param "
        "model (sam3.pt keyed detector.*/tracker.*, model.safetensors keyed "
        "detector_model.*/tracker_model.*/tracker_neck.*, zero shared keys). "
        "The native .pt is canonical here because the loader imports SAM 3's own "
        "python package by path rather than going through transformers, so the "
        "safetensors copy is the one to skip."
    )


def test_project_models_dir_is_found(bare, monkeypatch):
    """Where fetch_models.py installs things."""
    models = bare / "no-models"
    _hf_dir(models / registry.BY_ROLE["depth"].dir_name)
    monkeypatch.setattr(registry, "PROJECT_MODELS", models)
    assert registry.resolve("depth") == str(models / registry.BY_ROLE["depth"].dir_name)


def test_env_override_wins_over_the_project_dir(bare, monkeypatch):
    """For anyone who keeps a shared model directory outside the repo."""
    spec = registry.BY_ROLE["matte"]
    project = bare / "no-models"
    _hf_dir(project / spec.dir_name)
    monkeypatch.setattr(registry, "PROJECT_MODELS", project)

    elsewhere = bare / "elsewhere"
    _hf_dir(elsewhere / spec.dir_name)
    monkeypatch.setenv("HERON_MODELS", str(elsewhere))

    assert registry.resolve("matte") == str(elsewhere / spec.dir_name)


def test_a_directory_without_config_json_is_not_a_model(bare, monkeypatch):
    """An empty or half-downloaded folder must not count as installed."""
    models = bare / "no-models"
    (models / registry.BY_ROLE["depth"].dir_name).mkdir(parents=True)
    monkeypatch.setattr(registry, "PROJECT_MODELS", models)
    assert registry.resolve("depth") is None


def test_unknown_role_raises():
    with pytest.raises(KeyError, match="unknown model role"):
        registry.resolve("telepathy")


def test_status_covers_every_spec():
    assert len(registry.status()) == len(registry.SPECS)
    for spec, where in registry.status():
        assert where is None or isinstance(where, str)


def test_this_machine_needs_no_downloads():
    """CLAUDE.md §2.1: never download something already on this machine."""
    assert registry.missing() == [], (
        f"would re-download {[s.role for s in registry.missing()]} despite local copies"
    )


def test_engine_loaders_never_download():
    """local_files_only must stay on: the engine is offline and deterministic.

    Fetching is an explicit, separate step. A loader that can reach the network
    mid-render breaks both properties at once.
    """
    for name in ("depth.py", "matte.py"):
        text = (REPO / "heron" / "scene" / "ai" / name).read_text()
        assert "local_files_only=True" in text, f"{name} may download at render time"


def test_fetch_script_does_not_run_on_import():
    """Importing must never start a download; only main() may."""
    text = (REPO / "scripts" / "fetch_models.py").read_text()
    assert 'if __name__ == "__main__":' in text
    body = text.split('if __name__ == "__main__":')[0]
    assert "snapshot_download(" in body, "expected the call to exist"
    assert body.count("snapshot_download(\n") <= 1
