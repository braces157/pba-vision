"""Command line entry points.

    pba-vision videos                          what the official feed offers
    pba-vision find --season 2024 --keyword 월드 --players 사이그너 체네트
    pba-vision run --video ID --season 2024 --keyword 월드 --players A B --workdir data
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .align.innings import parse_set_innings
from .align.terminal import Terminal, TerminalError
from .ingest import download, extract_frames, media_videos
from .pipeline import reconstruct
from .publish import write_artifact
from .validate import validate


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="pba-vision", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    videos = sub.add_parser("videos", help="list broadcast sets from the official video feed")
    videos.add_argument("--pages", type=int, default=2)

    find = sub.add_parser("find", help="locate a match on the scoring terminal")
    find.add_argument("--season", type=int, required=True)
    find.add_argument("--keyword", required=True, help="title fragment, e.g. 월드챔피언십")
    find.add_argument("--players", nargs=2, required=True, help="two player names, as the Korean feed spells them")

    run = sub.add_parser("run", help="reconstruct one broadcast set end to end")
    run.add_argument("--video", required=True, help="YouTube video id of the set broadcast")
    run.add_argument("--season", type=int, required=True)
    run.add_argument("--keyword", required=True)
    run.add_argument("--players", nargs=2, required=True)
    run.add_argument("--set-number", type=int, default=None, help="which set; defaults to the last")
    run.add_argument("--workdir", type=Path, default=Path("data"))
    run.add_argument("--fps", type=float, default=4.0)
    run.add_argument("--out", type=Path, default=Path("reconstruction.json"))

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
    except (TerminalError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


def _run(args) -> int:
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

    sheet = parse_set_innings(terminal.set_innings_html(pcgm))
    print(f"official sheet: {sheet.players}, {sheet.inning_count} innings, final {sheet.final_totals()}")

    validation = validate(reconstruction, sheet)
    print(validation.summary())

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
    )
    print(f"artifact: {artifact}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
