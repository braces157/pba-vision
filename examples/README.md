# Example artifact

`scores_team_league_2026_s3.json` contains broadcast scores from Cenet vs
Q. Nguyen, PBA Team League 2026-27 Round 4, set 3. Video `jBbjq-VjxM4`
runs 18 minutes. The reader confirmed 0-0, 14 score changes, and the final
5-15 at 17:35.5. Each confirmed score was checked against a video frame.
Timestamps mark overlay updates and do not identify exact shot outcomes.

`history_team_league_2026_s3.json` preserves the eleven verified turns and
playing-bed calibration for the same video. Its movement arrays are empty
inputs, not a claim that the turns contain no shots. Regenerate measured
paths and static images with:

```bash
python scripts/trace_history.py examples/history_team_league_2026_s3.json \
  --video data/video_jBbjq-VjxM4.mp4 --out reports/match/history.json
python scripts/render_match_history.py reports/match/history.json --out reports/match/images
python scripts/review_history.py reports/match/history.json \
  --video data/video_jBbjq-VjxM4.mp4 --out reports/match/source_review
```

The last command overlays measurements on source frames for visual review.
Paths remain incomplete when the broadcast changes cameras. Turn totals
stay separate from individual movement outcomes.

`history_team_league_2026_s3_observed.json` contains the resulting measured
paths: 24 movements across all 11 turns. Render it directly with
`scripts/render_match_history.py`; source footage is only required to rescan
or review source frames. The matching `history_team_league_2026_s3_review.json`
records source metadata, review times and coverage limits.

[Browse the image gallery](../docs/match-history.md) for white- and yellow-cue
examples and links to all 24 full-size images.

Regenerate the score timeline:

```bash
pba-vision scores --video jBbjq-VjxM4 --players Cenet 'Q. Nguyen' \
  --set-number 3 --out data/scores_jBbjq-VjxM4.json
```

`reconstruction_wc2025_s5.json` — **SK Rent-a-Car Jeju PBA World
Championship 2025, final, set 5** (Saygıner vs Cenet), reconstructed from the
official broadcast VOD (`kxfju6anScQ`, 21 min, 1080p60) at 4 fps on CPU.

- match: PST 84099 (결승), set 5 = PCGM 164025 on the scoring terminal
- official result: 6 innings, Cenet 7 — Saygıner 15
- the `validation` block compares the CV reconstruction with that sheet;
  read the two known gaps in the root README before interpreting it

Plot it:

```bash
python scripts/plot_reconstruction.py reconstruction_wc2025_s5.json --out scatter.png
```
