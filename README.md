# pba-vision

Reconstruct **shot-by-shot ball positions for 3-cushion billiards** from PBA
(Korean Professional Billiards Association) broadcast recordings — and check
the reconstruction against the association's official scoring terminal.

The association publishes no ball-position data anywhere: its live terminal
records points per inning only, and the tracking graphics seen on broadcasts
are produced by the TV production and never leave the truck. This project
gets the data from the only source that has it — the video itself.

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
| full-table view / quad locked / all three balls | 3 374 / 2 378 / 2 183 |
| rest states → strokes | 142 → **88** |
| caroms | **23 detected vs 20 official** (+15%; official 22 points include four 2-point shots) |
| official result | 6 innings, Cenet 7 — Saygıner 15 |
| rest-position repeatability | ~5 mm median across consecutive frames |

Inning-level chaining is the one part that does **not** validate yet: rest
positions cannot tell a real miss from a camera cut reliably, and per-stroke
cue-ball identity stays ambiguous at 4 fps. Estimated innings (41) vs official
(6) is reported honestly in the artifact. Fixing this needs 60 fps burst
sampling around motion bursts or score-bug OCR — it is the top roadmap item.

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
