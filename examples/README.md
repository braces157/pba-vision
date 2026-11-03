# Example artifact

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
