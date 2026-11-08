# pba-vision

Reconstruct **shot-by-shot ball positions for 3-cushion billiards** from PBA
(Korean Professional Billiards Association) broadcast recordings — and check
the reconstruction against the association's official scoring terminal.

The association publishes no ball-position data anywhere: its live terminal
records points per inning only, and the tracking graphics seen on broadcasts
are produced by the TV production and never leave the truck. This project
gets the data from the only source that has it — the video itself.

## Match history in v0.3.0

See each player's turn with starting balls, measured travel, direction arrows,
numbered cushion estimates, and last-visible positions. Scores come from
the broadcast; a turn total is kept separate from individual movement outcomes.

![Nguyen's white-ball route with five cushion estimates and before/last-visible layouts](docs/images/team-league-s3/turn_07_movement_01.png)

[Browse all 24 images across 11 turns](docs/match-history.md) from
[Cenet vs Q. Nguyen, set 3](https://www.youtube.com/watch?v=jBbjq-VjxM4),
which finished **5–15**. The same tracker and renderer produced every image.
Three movements show a settled ending; the broadcast cuts away before the
other 21 settle. Missing footage stays blank and each cut ending is labeled.

Generate the image gallery from the bundled observations, without a video download:

```bash
pip install -e ".[plot]"
python scripts/render_match_history.py examples/history_team_league_2026_s3_observed.json --out reports/match/images
```

```
broadcast VOD ──▶ frame sampling ──▶ ball detection ──▶ cloth-quad homography
                                                        │ (positions in meters)
                                                        ▼
official inning sheet ◀── scoring terminal ── stroke & inning segmentation
        │
        └──▶ validation report + JSON artifact
```

## What it does, concretely

One command takes a broadcast set from YouTube to a validated artifact:

```bash
pba-vision run --video kxfju6anScQ --season 2024 --keyword 월드 \
               --players 사이그너 체네트 --workdir data --fps 4
```

- **VOD discovery** — reads the association's own media feed and scoring
  terminal (tournament → bracket → match → per-set codes). No channel
  scraping.
- **Detection** — the three balls are three distinct colours on a uniform
  cloth under studio lighting, so plain HSV segmentation with contour filters
  finds them without a learned model.
- **Geometry** — the cloth quad is detected per frame and mapped to the
  table's outer rectangle (≈3.10 × 1.70 m) with a homography. On PBA's
  overhead camera the mapping is near-affine, and rest positions repeat to
  ~5 mm median.
- **Strokes & innings** — stable multi-frame rest states become table states;
  transitions become strokes. Which balls moved classifies the outcome
  without needing cue-ball identity: the cue ball is the one struck, so
  one/two movers = miss, three movers = carom.
- **Validation** — the reconstruction is compared with the official inning
  sheet (innings, per-inning points, final totals) fetched from the scoring
  terminal. The report is printed, not massaged.

## Results — first full validated set

World Championship 2025 final (Saygıner vs Cenet), set 5, official broadcast
VOD, 21 minutes at 4 fps, CPU only:

<!-- RESULTS-BEGIN (see examples/ and docs/images for the artifact and plots) -->

| stage | result |
| --- | --- |
| frames read | 5 022 (21 min @ 4 fps, CPU only, no GPU) |
| camera survey | 28 views found, **1 stable (tripod) camera** owning 2 249 of 2 378 table frames |
| frames measured / all three balls | 1 830 on the locked camera |
| rest states → strokes | 132 → **81** |
| caroms | **15 detected vs 20 official** |
| official result | 6 innings, Cenet 7 — Saygıner 15 |
| rest-position repeatability | ~5 mm median across consecutive frames |

## The camera lock (v0.2)

A broadcast cuts constantly, and a cut is poison for position streams: the
same resting balls map a few centimetres apart on two different cameras, so
chaining rests across cuts *manufactures* movement — the v0.1 run overcounted
caroms (23 vs 20 official) exactly this way. The fix exploits a physical fact:
the scoring view is a **fixed tripod**, so its cloth quad is constant all
session. The survey pass clusters every plausible quad; the dominant cluster
is the tripod (here: 28 views found, one owning 95% of table time). Measurement
then accepts only frames on a stable camera, and strokes never span views.

The trade-off is measured, not hidden: shots that happen while only a *roaming*
camera is live have no stable quad, so they are skipped — hence 15 detected
caroms against 20 official. In exchange every position comes from one
calibrated view, and the phantom-carom bias disappears. Accepting roaming
cameras with per-frame homography is on the roadmap.

## Known, quantified gaps

Deliberate, not bugs — the validation output reports them per run:

- a carom that hits both object balls but completes fewer than three cushions
  scores nothing on the table, yet three moving balls imply one point;
- the terminal's 2-point shots are one carom each but two points on the sheet;
- shots covered only by roaming cameras are skipped (see the camera lock);
- per-stroke cue-ball identity is inferred and chaining is unreliable at 4 fps:
  estimated innings (42) vs official (6) is reported honestly in the artifact.
  Fixing it needs 60 fps burst sampling or score-bug OCR — the top roadmap item.

![reconstructed ball positions](docs/images/scatter.png)

Known, quantified gaps (deliberate, not bugs): a carom that hits both object
balls but completes fewer than three cushions scores nothing on the table but
reads as a point from rest positions alone; the terminal's 2-point shots are
one carom each but two points on the sheet. The validation output shows the
combined gap for every run.

## Install

Python 3.10+, [ffmpeg](https://ffmpeg.org) on PATH:

```bash
pip install -e .[dev]
pytest -q        # offline; synthetic frames, no fixtures from broadcasts
```

## Commands

### Read the broadcast score

For a Team League video, install the OCR extra and run `scores`:

```bash
pip install -e ".[scoreboard]"
pba-vision scores --video 'https://www.youtube.com/watch?v=jBbjq-VjxM4' \
  --players Cenet 'Q. Nguyen' --set-number 3 --out data/scores_jBbjq-VjxM4.json
```

This command reads the cumulative scores from the top and bottom score tiles.
It samples at 2 fps, requires two matching readings, and ignores backward
score changes from replays. It accepts a YouTube ID, a URL, or
`--local-video path/to/video.mp4`. It does not require the official terminal.
The layout profile supports the 16:9 PBA Team League score bug shown in this
video. For a set with a different winning score, pass `--target`.

The JSON includes confirmed score changes, timestamps, and the observed
final score. The [verified example](examples/scores_team_league_2026_s3.json)
records Cenet 5, Q. Nguyen 15, with 14 score changes checked against video
frames. Exit code 2 means the opening 0–0 or the winning score was not
confirmed. The file still contains the available observations. A confirmed
opening and winning score establish endpoint coverage; they do not prove
that every intermediate update was visible.

Score changes can lag the shot or cover several shots while the overlay is
hidden. They do not identify misses, cushion contacts, or the exact stroke
that earned each point. The reader ignores score decreases, including referee
corrections; review a video with a scoring correction manually.

To include the same score timeline in a `run` artifact, add
`--scoreboard-profile pba-team-league`. The `scoreboard` block contains
broadcast observations. The existing `points_est`, cue, and inning fields
remain estimates from ball positions.

### Render measured trajectories as images

Use a turn-history JSON with verified player turns, cue colours, cumulative
scores, and calibrated `bed_quad_960px` / `fixed_outer_quad_960px` corners:

```bash
python scripts/trace_history.py history.json --video local-video.mp4 --out reports/match/history.json
python scripts/render_match_history.py reports/match/history.json --out reports/match/images
```

The scanner checks every native video frame in each turn. It detects a
resting layout followed by cue motion, and keeps camera cuts and missing
detections as gaps. Returning footage with moving balls is labeled as a
fragment with an unseen start. The warm-ivory white-ball profile is calibrated
for this Team League broadcast at 960×540; other camera layouts and ball
colours require calibration and source-video review.

The renderer writes a PNG for each observed movement, a card for any turn
without a path, and an `index.json` covering every turn. Every image uses
the same geometry, colours, arrows, initial balls, last-visible markers,
and numbered cushion estimates. Two frames can bracket a cushion estimate
across a short gap, labeled `bracketed_gap` in JSON; the drawn path still
remains broken. Cushion estimates do not establish a legal scoring shot.

Turn totals come from the broadcast. They are never assigned to an individual
movement automatically. A movement is not necessarily a whole shot, and
an automatically tracked path still needs review against its source footage.
No browser or inline visualization runtime is needed to view the PNGs.

### Reconstruct ball positions

```bash
pba-vision videos                      # what the official feed currently offers
pba-vision find --season 2024 --keyword 월드 --players 사이그너 체네트
pba-vision run   --video kxfju6anScQ --season 2024 --keyword 월드 \
                 --players 사이그너 체네트 --workdir data --fps 4 \
                 --out reconstruction.json
python scripts/plot_reconstruction.py reconstruction.json --out scatter.png
```

The JSON artifact (`schema: pba-vision/reconstruction@0.1`) carries per-stroke
from/to positions in meters, moved-ball flags, estimated outcome and inning,
plus the official-vs-estimated validation block. A curated example ships in
[`examples/`](examples/).

## Layout

```
src/pbavision/
  ingest.py      media feed, yt-dlp download, ffmpeg frame extraction
  vision/
    detect.py    HSV ball detection + full-table gate
    camera.py    stable-camera discovery and the per-camera lock
    table.py     cloth quad, homography to table meters
  align/
    terminal.py  scoring-terminal client (TLS intermediate pinned, referer set)
    innings.py   official inning sheet parser
  pipeline.py    rest states → strokes → innings
  validate.py    reconstruction vs official sheet
  publish.py     the JSON artifact
```

## Honest limitations

- **Scoring is approximate.** Rest positions cannot see cushion counts, so
  three-cushion-rule failures overcount and 2-point shots undercount. The
  validation report quantifies this per run.
- **Cue-ball identity is inferred**, anchored by single-mover strokes and
  chained by the score/miss rule; until the first anchor it is unknown.
- **Overlays bite.** Score bugs and broadcast graphics occasionally produce
  false candidates; the table gate and per-colour best-candidate selection
  absorb most of it, not all.
- The classifier is 85% of the way with zero ML. The remaining 20% (a tiny
  YOLO detector, bed-edge refinement, 60 fps burst sampling for cushion
  counting) is the roadmap.

## Legal

Unofficial, unaffiliated with the KPA or PBA TV. Video is downloaded with
yt-dlp for personal analysis; no footage or frames are committed to this
repository. The bundled TLS certificate is a public GlobalSign intermediate.

## License

[MIT](LICENSE)
