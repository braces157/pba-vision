"""Reconstruct shot-by-shot ball positions from PBA broadcast recordings.

The association publishes no ball-position data: its scoring terminal records
points per inning only. This package reads the broadcast video instead —
detect the three balls, map them to table coordinates with a cloth-quad
homography, segment strokes from rest positions, and check the resulting
score reconstruction against the official inning sheet.
"""

__version__ = "0.2.0"
