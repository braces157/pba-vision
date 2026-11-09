import importlib
import json

import numpy as np
import pytest


@pytest.fixture
def tools(monkeypatch):
    from pathlib import Path

    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    return importlib.import_module("trace_history"), importlib.import_module("review_history")


def history(turns=None):
    return {"schema": "pba-vision/turn-history@0.3", "turns": turns or [],
            "calibration": {"bed_quad_960px": [[10, 10], [950, 10], [950, 530], [10, 530]],
                            "fixed_outer_quad_960px": [[5, 5], [955, 5], [955, 535], [5, 535]]}}


def test_empty_history_creates_output_parent(tools, tmp_path):
    trace, _ = tools
    source, out = tmp_path / "input.json", tmp_path / "new" / "history.json"
    source.write_text(json.dumps(history()))
    trace.main([str(source), "--video", "unused.mp4", "--out", str(out)])
    assert json.loads(out.read_text())["scan"]["complete"] is True


def test_unknown_turn_is_an_explicit_input_error(tools, tmp_path):
    trace, _ = tools
    source, out = tmp_path / "input.json", tmp_path / "new" / "history.json"
    source.write_text(json.dumps(history()))
    with pytest.raises(SystemExit) as error:
        trace.main([str(source), "--turn", "7", "--video", "unused.mp4", "--out", str(out)])
    assert error.value.code == 2
    assert not out.exists()


def test_partial_rescan_rejects_untraced_mixed_geometry(tools, tmp_path):
    trace, _ = tools
    source = tmp_path / "input.json"
    data = history([{"turn": 1, "movements": []}, {"turn": 2, "movements": [{"from": {}}]}])
    data["scan"] = {"completed_turns": [1, 2]}
    source.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="fully traced"):
        trace.main([str(source), "--turn", "1", "--video", "unused.mp4", "--out", str(tmp_path / "out.json")])


def test_full_scan_omits_unconverted_outer_layouts(tools, tmp_path, monkeypatch):
    trace, _ = tools
    data = history([{"turn": 1, "movements": [], "layout": {"white": [3., 1.6]}}])
    data["schema"] = "pba-vision/turn-history@0.1"
    source, out = tmp_path / "input.json", tmp_path / "out.json"
    source.write_text(json.dumps(data))
    monkeypatch.setattr(trace, "scan_turn", lambda *args: [])
    trace.main([str(source), "--video", "unused.mp4", "--out", str(out)])
    result = json.loads(out.read_text())
    assert result["turns"][0]["layout"] is None
    assert result["scan"]["completed_turns"] == [1]


def test_review_crop_contains_calibrated_edges_outside_old_crop(tools):
    _, review = tools
    bed = np.array([[10, 10], [950, 10], [950, 530], [10, 530]], np.float32)
    assert review.review_crop(bed, (540, 960)) == (0, 0, 960, 540)
