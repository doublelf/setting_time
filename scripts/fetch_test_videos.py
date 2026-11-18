"""Download the public test videos used to validate the pipeline.

Sources are Mixkit free stock footage (Mixkit Free License: free for commercial use,
no attribution required) and Pexels stock footage.
"""

from __future__ import annotations

import sys
from pathlib import Path

import urllib.request

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "assets" / "videos"

UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"

CLIPS = [
    # (local name, url, why it is in the corpus)
    (
        "coffee_shop.mp4",
        "https://assets.mixkit.co/videos/4350/4350-1080.mp4",
        "1920x1080 indoor cafe, ~6 people per frame: the main multi-person occlusion test",
    ),
    (
        "mall_busy_timelapse.mp4",
        "https://assets.mixkit.co/videos/31086/31086-720.mp4",
        "dense crowd, accelerated motion: stress test for the tracker",
    ),
    (
        "supermarket_browse.mp4",
        "https://assets.mixkit.co/videos/49137/49137-720.mp4",
        "person browsing shelves: a genuine long dwell",
    ),
    (
        "lobby_static.mp4",
        "https://assets.mixkit.co/videos/27896/27896-720.mp4",
        "person standing still for ~29 s: timer accuracy baseline",
    ),
    (
        "fashion_mall.mp4",
        "https://assets.mixkit.co/videos/9060/9060-720.mp4",
        "several shoppers in a fashion mall: secondary multi-person clip",
    ),
    (
        "mall_walk.mp4",
        "https://assets.mixkit.co/videos/25516/25516-720.mp4",
        "shoppers walking through a mall: entry/exit behaviour",
    ),
]


def download(url: str, target: Path) -> bool:
    """Fetch a URL into `target`, skipping the work if it is already there."""
    if target.exists() and target.stat().st_size > 100_000:
        print(f"  skip  {target.name} (already present)")
        return True
    request = urllib.request.Request(url, headers={"User-Agent": UA})
    try:
        with urllib.request.urlopen(request, timeout=180) as response, open(target, "wb") as fh:
            fh.write(response.read())
    except Exception as exc:  # noqa: BLE001 - report and continue with the other clips
        print(f"  FAIL  {target.name}: {exc}")
        return False
    print(f"  ok    {target.name}  {target.stat().st_size / 1e6:.1f} MB")
    return True


def main() -> int:
    """Download every clip of the test corpus."""
    DEST.mkdir(parents=True, exist_ok=True)
    print(f"downloading test clips into {DEST}")
    ok = True
    for name, url, note in CLIPS:
        print(f"{name}: {note}")
        ok &= download(url, DEST / name)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
