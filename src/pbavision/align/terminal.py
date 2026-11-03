"""Client for the association's hosts.

Two hosts carry the data this package aligns against:

    www.pbatour.org   JSON API — tournament lists (which TourCode is which)
    app.pbatour.org   legacy ASP scoring terminal — brackets, set codes,
                      inning sheets; HTML fragments

Both serve their TLS leaf without the intermediate certificate, so every
client fails chain building unless the GlobalSign intermediate is pinned.
It ships alongside this module (globalsign_gccr46_alphasslca2025.pem, a
public CA certificate) and is added to the default trust store, exactly as
the Node client of the companion scoreboard site does.

The terminal also rejects requests without a browser-like Referer pointing
at its own live board, which every request below supplies.
"""

from __future__ import annotations

import json
import re
import ssl
import urllib.error
import urllib.request
from pathlib import Path

SITE_ORIGIN = "https://www.pbatour.org"
APP_ORIGIN = "https://app.pbatour.org"
LIVE_BOARD_PATH = "/mob/contest_new/admin_en/live_info.asp"

_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/131.0 Safari/537.36"
)
_PEM = Path(__file__).with_name("globalsign_gccr46_alphasslca2025.pem").read_text(encoding="utf-8")

_CONTEXT = ssl.create_default_context()
_CONTEXT.load_verify_locations(cadata=_PEM)


class TerminalError(RuntimeError):
    """A request to the association failed or returned unusable data."""


class Terminal:
    """Read-only access to the feeds the reconstruction is validated against."""

    def __init__(self, timeout: int = 25) -> None:
        self.timeout = timeout

    def _get(self, url: str, referer: str) -> str:
        request = urllib.request.Request(
            url,
            headers={
                "User-Agent": _UA,
                "Accept": "text/html, */*" if ".asp" in url else "application/json, text/plain, */*",
                "Accept-Language": "ko,ko-KR;q=0.9,en;q=0.8",
                "Referer": referer,
                "X-Requested-With": "XMLHttpRequest",
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout, context=_CONTEXT) as response:
                return response.read().decode("utf-8", errors="replace")
        except (urllib.error.URLError, OSError) as exc:
            raise TerminalError(f"{url} failed: {exc}") from exc

    def _site(self, path: str) -> str:
        return self._get(f"{SITE_ORIGIN}{path}", f"{SITE_ORIGIN}/en/index")

    def _terminal(self, path: str) -> str:
        return self._get(f"{APP_ORIGIN}{path}", f"{APP_ORIGIN}{LIVE_BOARD_PATH}")

    # ------------------------------------------------------------- tournaments

    def find_tournament(self, season: int, keyword: str) -> dict:
        """The tournament of ``season`` whose title contains ``keyword``.

        Season years run August-May; the SK Rent-a-Car 2025 championship lives
        in the 2024 list. Raises :class:`TerminalError` when nothing matches.
        """
        body = self._site(f"/en/tournament/schedule/ajax/list?langCode=ko&season={season}")
        rows = json.loads(body).get("list") or []
        for row in rows:
            if keyword.lower() in f"{row.get('TITLE', '')} {row.get('TITLE_EN', '')}".lower():
                return {
                    "seq": row["SEQ"],
                    "tour_code": int(row["MATCH_CODE"]),
                    "title": row.get("TITLE_EN") or row["TITLE"],
                }
        raise TerminalError(f"no tournament in season {season} matches {keyword!r}")

    # ----------------------------------------------------------------- bracket

    def find_match(self, tour_code: int, names: list[str]) -> dict:
        """The match whose row names every player in ``names``.

        Bracket pages come 20 rows to a page, one page per bracket size is
        enough for finals and semifinals; the search walks sizes from the
        final (2) outward. Returns the match code and the round label.
        """
        for size in (2, 4, 8, 16, 32, 64, 128):
            html = self._terminal(
                f"/mob/contest_new/code_behind/contest_result_sur.asp"
                f"?TourCode={tour_code}&iBLCode={size}&iPage=1&txt_Search=&LangCode=ko"
            )
            for pst_code, chunk in _bracket_blocks(html):
                if all(name in chunk for name in names):
                    round_label = _round_label(html) or f"bracket {size}"
                    return {"pst_code": int(pst_code), "round": round_label}
        raise TerminalError(f"no bracket match in tour {tour_code} names every of {names}")

    def set_codes(self, pst_code: int) -> list[tuple[int, int]]:
        """(PCGMCode, set number) pairs of a match, in set order."""
        html = self._terminal(f"/mob/contest_new/admin_en/live_info_setview.asp?PSTCode={pst_code}&LangCode=ko")
        match = re.search(r'var\s+gameNumber\s*=\s*"([^"]*)"', html)
        if not match:
            raise TerminalError(f"set view for PST {pst_code} carries no set list")
        codes: list[tuple[int, int]] = []
        for part in match.group(1).split(","):
            fields = part.split("|")
            if len(fields) >= 2 and fields[0].isdigit() and fields[1].isdigit():
                codes.append((int(fields[0]), int(fields[1])))
        if not codes:
            raise TerminalError(f"set view for PST {pst_code} has an empty set list")
        return codes

    def set_innings_html(self, pcgm_code: int) -> str:
        """The inning sheet fragment — one row per inning, both players' columns."""
        return self._terminal(
            f"/mob/contest_new/code_behind/contest_result_set_record.asp?PCGMCode={pcgm_code}&LangCode=ko"
        )


def _bracket_blocks(html: str) -> list[tuple[str, str]]:
    """(PSTCode, surrounding markup) for every setview() call on a bracket page."""
    blocks = []
    for match in re.finditer(r"setview\(\s*(\d+)\s*\)", html):
        start = max(0, match.start() - 400)
        # The names sit in the result block below the row's button, up to
        # ~2.5 KB after it, so the context window reaches forward generously.
        blocks.append((match.group(1), html[start:match.end() + 2600]))
    return blocks


def _round_label(chunk: str) -> str | None:
    match = re.search(r"<strong[^>]*>([^<]*(?:Round|결승|라운드)[^<]*)</strong>", chunk)
    return match.group(1).strip() if match else None
