"""
HTTP client for the Battlesnake Blackout competition site.

Replay JSON: ``GET {base}/api/replay/{game_id}``
Leaderboard and recent games are scraped from public HTML pages.
"""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

BLACKOUT_BASE_URL = "https://www.tnt.uni-hannover.de/bs-blackout-2026"
_USER_AGENT = "battle-snake-local/1.0"


@dataclass(frozen=True)
class LeaderboardEntry:
    snake_id: int
    name: str
    rating: Optional[float]


def _get(url: str, timeout: float = 30.0) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read().decode("utf-8", errors="replace")


def fetch_replay(game_id: int | str, *, base_url: str = BLACKOUT_BASE_URL) -> Dict[str, Any]:
    url = f"{base_url.rstrip('/')}/api/replay/{game_id}"
    req = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=60.0) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"Blackout replay {game_id} failed: HTTP {e.code}") from e
    except json.JSONDecodeError as e:
        raise RuntimeError(f"Blackout replay {game_id}: invalid JSON") from e


def fetch_leaderboard(*, base_url: str = BLACKOUT_BASE_URL) -> List[LeaderboardEntry]:
    html = _get(f"{base_url.rstrip('/')}/")
    rows = re.findall(
        r"<tr>.*?/snake/(\d+)[^>]*>([^<]+)</a>.*?<td[^>]*>([^<]+)</td>",
        html,
        flags=re.DOTALL,
    )
    out: List[LeaderboardEntry] = []
    for sid, name, rating_s in rows:
        rating_s = rating_s.strip()
        rating = None if rating_s.lower() == "unranked" else float(rating_s)
        out.append(LeaderboardEntry(snake_id=int(sid), name=name.strip(), rating=rating))
    return out


def fetch_recent_game_ids(
    snake_id: int,
    *,
    limit: int = 20,
    base_url: str = BLACKOUT_BASE_URL,
) -> List[int]:
    html = _get(f"{base_url.rstrip('/')}/snake/{snake_id}")
    ids = re.findall(r"/bs-blackout-2026/game/(\d+)", html)
    seen: set[str] = set()
    ordered: List[int] = []
    for gid in ids:
        if gid in seen:
            continue
        seen.add(gid)
        ordered.append(int(gid))
        if len(ordered) >= limit:
            break
    return ordered


def fetch_featured_game_id(*, base_url: str = BLACKOUT_BASE_URL) -> Optional[int]:
    html = _get(f"{base_url.rstrip('/')}/")
    m = re.search(r"api/replay/(\d+)", html)
    return int(m.group(1)) if m else None


def sample_leaderboard_game_ids(
    *,
    games_per_snake: int = 3,
    max_snakes: int = 5,
    base_url: str = BLACKOUT_BASE_URL,
) -> List[Tuple[int, str]]:
    """
    Return ``(game_id, label)`` pairs from top leaderboard snakes' recent games.
    """
    board = fetch_leaderboard(base_url=base_url)[:max_snakes]
    pool: List[Tuple[int, str]] = []
    seen_ids: set[int] = set()
    for entry in board:
        for gid in fetch_recent_game_ids(entry.snake_id, limit=games_per_snake, base_url=base_url):
            if gid in seen_ids:
                continue
            seen_ids.add(gid)
            pool.append((gid, f"{entry.name} #{gid}"))
    return pool
