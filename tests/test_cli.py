import json
from types import SimpleNamespace

from pbavision.cli import main
from pbavision.pipeline import Reconstruction, RestState, Stroke


def test_run_exports_raw_rest_intervals_and_uses_requested_score_target(tmp_path, monkeypatch):
    import pbavision.cli as cli

    video_id = "jBbjq-VjxM4"
    (tmp_path / f"video_{video_id}.mp4").touch()
    frames = tmp_path / f"frames_{video_id}"
    frames.mkdir()
    (frames / "f_000001.jpg").touch()
    before = RestState(0, 0, {"white": (.5, .5), "yellow": (1., 1.), "red": (2., 1.)}, 3, .75)
    after = RestState(8, 2, {**before.positions, "white": (1.5, .5)}, 11, 2.75)
    stroke = Stroke(0, before, after, {"white": True, "yellow": False, "red": False}, 0, "white", "white", 1)
    raw = Reconstruction(rests=[before, after], strokes=[stroke])
    terminal = SimpleNamespace(find_tournament=lambda *a: {"tour_code": 1, "title": "Test"},
                               find_match=lambda *a: {"pst_code": 1, "round": "Final"},
                               set_codes=lambda *a: [(1, 1)], set_innings_html=lambda *a: "sheet")
    monkeypatch.setattr(cli, "Terminal", lambda: terminal)
    monkeypatch.setattr(cli, "reconstruct", lambda *a, **k: raw)
    sheet = SimpleNamespace(players=["A", "B"], inning_count=1, final_totals=lambda: [0, 11])
    monkeypatch.setattr(cli, "parse_set_innings", lambda *a: sheet)
    monkeypatch.setattr(cli, "validate", lambda *a: SimpleNamespace(summary=lambda: "Validated"))
    targets, published = [], []

    def read_scores(*args, target, **kwargs):
        targets.append(target)
        return SimpleNamespace(to_dict=lambda: {"target": target})

    def write(path, **kwargs):
        published.append(kwargs)
        return path

    monkeypatch.setattr(cli, "read_video_scores", read_scores)
    monkeypatch.setattr(cli, "write_artifact", write)
    output = tmp_path / "raw" / "rests.json"
    assert main(["run", "--video", video_id, "--season", "2026", "--keyword", "test",
                 "--players", "A", "B", "--workdir", str(tmp_path), "--raw-out", str(output),
                 "--out", str(tmp_path / "published.json"), "--scoreboard-profile", "pba-team-league",
                 "--scoreboard-target", "11"]) == 0
    result = json.loads(output.read_text())
    assert result["rests"][0]["end_timestamp"] == .75
    assert result["strokes"][0]["from_rest"] == result["rests"][0]
    assert targets == [11]
    assert published[0]["reconstruction"] is raw
    assert published[0]["broadcast_scores"] == {"target": 11}


def test_raw_and_published_outputs_cannot_overwrite_one_another(tmp_path):
    output = str(tmp_path / "output.json")
    assert main(["run", "--video", "jBbjq-VjxM4", "--season", "2026", "--keyword", "test",
                 "--players", "A", "B", "--out", output, "--raw-out", output]) == 1
