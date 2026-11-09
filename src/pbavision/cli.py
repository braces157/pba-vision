"""Command line entry points.

    pba-vision videos                          what the official feed offers
    pba-vision find --season 2024 --keyword 월드 --players 사이그너 체네트
    pba-vision run --video ID --season 2024 --keyword 월드 --players A B --workdir data
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

from .align.innings import parse_set_innings
from .align.terminal import Terminal, TerminalError
from .ingest import download, extract_frames, media_videos, youtube_id
from .pipeline import reconstruct
from .publish import write_artifact
from .validate import validate
from .vision.scoreboard import read_video_scores


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="pba-vision", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    videos = sub.add_parser("videos", help="list broadcast sets from the official video feed")
    videos.add_argument("--pages", type=int, default=2)

    find = sub.add_parser("find", help="locate a match on the scoring terminal")
    find.add_argument("--season", type=int, required=True)
    find.add_argument("--keyword", required=True, help="title fragment, e.g. 월드챔피언십")
    find.add_argument("--players", nargs=2, required=True, help="two player names, as the Korean feed spells them")

    scores = sub.add_parser("scores", help="read real scores from a PBA Team League broadcast")
    scores.add_argument("--video", help="YouTube video ID or watch URL")
    scores.add_argument("--local-video", type=Path, help="read an existing video instead of downloading")
    scores.add_argument("--workdir", type=Path, default=Path("data"))
    scores.add_argument("--sample-fps", type=float, default=2.0)
    scores.add_argument("--target", type=int, default=15, help="winning set score")
    scores.add_argument("--players", nargs=2, help="player names in scoreboard order: top then bottom")
    scores.add_argument("--set-number", type=int)
    scores.add_argument("--out", type=Path, default=Path("scores.json"))

    run = sub.add_parser("run", help="reconstruct one broadcast set end to end")
    run.add_argument("--video", required=True, help="YouTube video id of the set broadcast")
    run.add_argument("--season", type=int, required=True)
    run.add_argument("--keyword", required=True)
    run.add_argument("--players", nargs=2, required=True)
    run.add_argument("--set-number", type=int, default=None, help="which set; defaults to the last")
    run.add_argument("--workdir", type=Path, default=Path("data"))
    run.add_argument("--fps", type=float, default=4.0)
    run.add_argument("--out", type=Path, default=Path("reconstruction.json"))
    run.add_argument("--raw-out", type=Path, help="also export rests and raw stroke intervals for plotting")
    run.add_argument("--scoreboard-profile", choices=["pba-team-league"],
                     help="include broadcast scores separately from CV estimates")
    run.add_argument("--scoreboard-target", type=int, default=15, help="winning score for the broadcast timeline")

    args = parser.parse_args(argv)
    try:
        if args.command == "videos":
            for video in media_videos(page_size=24 * args.pages):
                print(json.dumps(video, ensure_ascii=False))
            return 0

        if args.command == "find":
            terminal = Terminal()
            tournament = terminal.find_tournament(args.season, args.keyword)
            match = terminal.find_match(tournament["tour_code"], args.players)
            sets = terminal.set_codes(match["pst_code"])
            print(json.dumps({"tournament": tournament, "match": match, "sets": sets}, ensure_ascii=False, indent=2))
            return 0

        if args.command == "run":
            return _run(args)
        if args.command == "scores":
            return _scores(args)
    except (TerminalError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


def _run(args) -> int:
    if args.raw_out and args.raw_out.resolve() == args.out.resolve():
        raise ValueError("--raw-out must differ from the published --out path")
    terminal = Terminal()
    tournament = terminal.find_tournament(args.season, args.keyword)
    match = terminal.find_match(tournament["tour_code"], args.players)
    sets = terminal.set_codes(match["pst_code"])
    set_no = args.set_number if args.set_number else sets[-1][1]
    pcgm = next(code for code, no in sets if no == set_no)

    print(f"tour {tournament['tour_code']} ({tournament['title']}), "
          f"PST {match['pst_code']} ({match['round']}), set {set_no} -> PCGM {pcgm}")

    video_path = args.workdir / f"video_{args.video}.mp4"
    if video_path.exists():
        print(f"video cached: {video_path}")
    else:
        print(f"downloading {args.video} ...")
        download(args.video, video_path)

    frames_dir = args.workdir / f"frames_{args.video}"
    frames = sorted(frames_dir.glob("f_*.jpg")) if frames_dir.exists() else []
    if not frames:
        print(f"extracting frames at {args.fps} fps ...")
        frames = extract_frames(video_path, args.fps, frames_dir)
    print(f"{len(frames)} frames")

    reconstruction = reconstruct(frames, args.fps, progress=lambda msg: print(f"  {msg}"))
    if args.raw_out:
        args.raw_out.parent.mkdir(parents=True, exist_ok=True)
        raw = {"schema": "pba-vision/raw-reconstruction@0.1", **asdict(reconstruction)}
        args.raw_out.write_text(json.dumps(raw, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"raw reconstruction: {args.raw_out}")

    sheet = parse_set_innings(terminal.set_innings_html(pcgm))
    print(f"official sheet: {sheet.players}, {sheet.inning_count} innings, final {sheet.final_totals()}")

    validation = validate(reconstruction, sheet)
    print(validation.summary())

    broadcast_scores = None
    if args.scoreboard_profile:
        broadcast_scores = read_video_scores(video_path, target=args.scoreboard_target, progress=print).to_dict()

    artifact = write_artifact(
        args.out,
        meta={
            "video_id": args.video,
            "season": args.season,
            "tournament": tournament["title"],
            "tour_code": tournament["tour_code"],
            "pst_code": match["pst_code"],
            "pcgm_code": pcgm,
            "set_number": set_no,
            "players_official": sheet.players,
            "fps": args.fps,
        },
        reconstruction=reconstruction,
        validation=validation,
        broadcast_scores=broadcast_scores,
    )
    print(f"artifact: {artifact}")
    return 0


def _scores(args) -> int:
    if not args.video and not args.local_video:
        raise ValueError("provide --video or --local-video")
    video_id = youtube_id(args.video) if args.video else None
    path = args.local_video or args.workdir / f"video_{video_id}.mp4"
    if not args.local_video and not path.exists():
        download(video_id, path)
    timeline = read_video_scores(path, sample_fps=args.sample_fps, target=args.target, progress=print)
    artifact = {
        "schema": "pba-vision/scores@0.1",
        "meta": {"video_id": video_id, "video_path": str(path), "sample_fps": args.sample_fps,
                 "players": args.players, "set_number": args.set_number},
        "scoreboard": timeline.to_dict(),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(artifact, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"broadcast final (top, bottom): {artifact['scoreboard']['final']}")
    print(f"artifact: {args.out}")
    if not artifact["scoreboard"]["starts_at_zero"] or not artifact["scoreboard"]["reaches_target"]:
        print("incomplete set coverage: could not confirm both opening 0-0 and a winning score", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
