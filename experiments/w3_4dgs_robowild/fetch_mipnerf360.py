"""Stream-extract selected Mip-NeRF 360 scenes from Inria's pretrained models.zip via HTTP Range."""

from __future__ import annotations

import argparse
import io
import os
import time
import urllib.request
import zipfile


MODELS_URL = "https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/datasets/pretrained/models.zip"


class HttpRangeFile(io.RawIOBase):
  """Read-only seekable file backed by HTTP Range requests."""

  def __init__(self, url: str, chunk_size: int = 8 * 1024 * 1024):
    self.url = url
    self.chunk_size = chunk_size
    req = urllib.request.Request(url, method="HEAD")
    with urllib.request.urlopen(req, timeout=30) as resp:
      self.size = int(resp.headers["Content-Length"])
    self.pos = 0
    self.buf_start = -1
    self.buf = b""

  def readable(self) -> bool:
    return True

  def seekable(self) -> bool:
    return True

  def tell(self) -> int:
    return self.pos

  def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
    if whence == io.SEEK_SET:
      new_pos = offset
    elif whence == io.SEEK_CUR:
      new_pos = self.pos + offset
    elif whence == io.SEEK_END:
      new_pos = self.size + offset
    else:
      raise ValueError(f"Unsupported whence={whence}")
    self.pos = max(0, min(self.size, new_pos))
    return self.pos

  def _fetch(self, start: int, end: int) -> bytes:
    req = urllib.request.Request(self.url, headers={"Range": f"bytes={start}-{end}"})
    with urllib.request.urlopen(req, timeout=60) as resp:
      return resp.read()

  def readinto(self, b) -> int:
    if self.pos >= self.size:
      return 0
    n = min(len(b), self.size - self.pos)
    if not (self.buf_start <= self.pos and self.pos + n <= self.buf_start + len(self.buf)):
      fetch_len = max(n, self.chunk_size)
      end = min(self.size - 1, self.pos + fetch_len - 1)
      self.buf_start = self.pos
      self.buf = self._fetch(self.pos, end)
    off = self.pos - self.buf_start
    take = min(n, len(self.buf) - off)
    b[:take] = self.buf[off : off + take]
    self.pos += take
    return take


def fetch_scenes(out_dir: str, scenes: tuple[str, ...]) -> None:
  os.makedirs(out_dir, exist_ok=True)
  targets = []
  for sc in scenes:
    targets.append(f"{sc}/point_cloud/iteration_30000/point_cloud.ply")
    targets.append(f"{sc}/cameras.json")

  missing = [t for t in targets if not os.path.exists(os.path.join(out_dir, t))]
  if not missing:
    print(f"All targets already exist in {out_dir}", flush=True)
    return

  t0 = time.perf_counter()
  rf = io.BufferedReader(HttpRangeFile(MODELS_URL), buffer_size=8 * 1024 * 1024)
  with zipfile.ZipFile(rf, "r") as zf:
    names = set(zf.namelist())
    for rel in missing:
      # Match either exact path or prefix inside zip
      match = rel if rel in names else next((n for n in names if n.endswith("/" + rel) or n == rel), None)
      if match is None:
        raise FileNotFoundError(f"{rel} not found in models.zip")
      dst = os.path.join(out_dir, rel)
      os.makedirs(os.path.dirname(dst), exist_ok=True)
      t1 = time.perf_counter()
      info = zf.getinfo(match)
      print(f"Fetching {match} ({info.file_size / 1e6:.1f} MB) -> {dst} ...", flush=True)
      with zf.open(info, "r") as src, open(dst, "wb") as out_f:
        while True:
          blk = src.read(4 * 1024 * 1024)
          if not blk:
            break
          out_f.write(blk)
      print(f"  done in {time.perf_counter() - t1:.1f}s", flush=True)
  print(f"Total fetch time: {time.perf_counter() - t0:.1f}s", flush=True)


def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("--out_dir", default="/dev/shm/zaychik-w3-v6/data")
  ap.add_argument("--scenes", nargs="+", default=["kitchen", "garden"])
  args = ap.parse_args()
  fetch_scenes(args.out_dir, tuple(args.scenes))


if __name__ == "__main__":
  main()
