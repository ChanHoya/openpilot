"""Wi-Fi incremental sync client for Carrot Web device endpoints and golden corpus management."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from datetime import datetime
import ipaddress
import json
import os
from pathlib import Path
import re
import shutil
import socket
import subprocess
from typing import Any, Sequence
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlparse
from urllib.request import Request, urlopen

from . import (
  CONFIG_PATH,
  DEFAULT_CANDIDATE_HOSTS,
  DEFAULT_PORT,
  DEFAULT_TIMEOUT_S,
  GOLDEN_DIR,
  REPORTS_DIR,
  ROUTES_DIR,
  SCANS_DIR,
  ensure_analysis_dirs,
)

_FILENAME_RE = re.compile(r'filename="?([^";]+)"?')
_SAFE_NAME_RE = re.compile(r"[^A-Za-z0-9._-]+")


@dataclass
class SyncResult:
  host: str
  port: int
  device_info: dict[str, Any] = field(default_factory=dict)
  reports_saved: int = 0
  reports_skipped: int = 0
  routes_synced: list[str] = field(default_factory=list)
  segments_downloaded: int = 0
  segments_skipped: int = 0
  bytes_downloaded: int = 0
  downloaded_files: list[str] = field(default_factory=list)
  errors: list[str] = field(default_factory=list)

  def to_dict(self) -> dict[str, Any]:
    return asdict(self)


@dataclass
class RouteGroup:
  route: str
  source: str = "device"  # "device+local" | "device" | "local"
  device_segments: int = 0
  local_segments: int = 0
  started_at: str = ""
  ended_at: str = ""
  start_epoch: float = 0.0
  end_epoch: float = 0.0
  duration_min: float = 0.0
  latest_label: str = ""
  local_dir: str = ""
  matched_report: dict[str, Any] | None = None
  analyzed_html: str = ""

  def to_dict(self) -> dict[str, Any]:
    return asdict(self)


def load_config(config_path: Path = CONFIG_PATH) -> dict[str, Any]:
  if not config_path.is_file():
    return {}
  try:
    data = json.loads(config_path.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else {}
  except Exception:
    return {}


def extract_dongle_id(device_info: dict[str, Any] | None) -> str:
  if not isinstance(device_info, dict):
    return ""
  identity = device_info.get("identity")
  if isinstance(identity, dict) and identity.get("dongle_id"):
    return str(identity["dongle_id"]).strip().lower()
  for key in ("dongle_id", "DongleId", "dongleId"):
    if device_info.get(key):
      return str(device_info[key]).strip().lower()
  return ""


def save_config(
  host: str | None = None,
  port: int | None = None,
  timeout_s: float | None = None,
  dongle_id: str | None = None,
  *,
  clear_dongle_id: bool = False,
  clear_host: bool = False,
  config_path: Path = CONFIG_PATH,
) -> dict[str, Any]:
  existing = load_config(config_path)
  merged_port = int(port if port is not None else (existing.get("port") or DEFAULT_PORT))
  merged_timeout = float(timeout_s if timeout_s is not None else (existing.get("timeout_s") or DEFAULT_TIMEOUT_S))

  payload: dict[str, Any] = dict(existing)
  payload["port"] = merged_port
  payload["timeout_s"] = merged_timeout

  if clear_host:
    payload.pop("host", None)
  elif host is not None and host.strip():
    clean_host, parsed_port = _normalize_host_and_port(host, merged_port)
    payload["host"] = clean_host
    payload["port"] = parsed_port

  if clear_dongle_id:
    payload.pop("dongle_id", None)
  elif dongle_id is not None and dongle_id.strip():
    payload["dongle_id"] = dongle_id.strip().lower()

  config_path.parent.mkdir(parents=True, exist_ok=True)
  config_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
  return payload


def probe_and_save_host(
  host: str,
  port: int | None = None,
  timeout_s: float | None = None,
  dongle_id: str | None = None,
  *,
  config_path: Path = CONFIG_PATH,
) -> dict[str, Any]:
  """Save host and automatically probe & persist the device's dongle_id when reachable."""
  cfg = load_config(config_path)
  resolved_port = int(port if port is not None else (cfg.get("port") or DEFAULT_PORT))
  resolved_timeout = float(timeout_s if timeout_s is not None else (cfg.get("timeout_s") or DEFAULT_TIMEOUT_S))
  clean_host, parsed_port = _normalize_host_and_port(host, resolved_port)
  detected_id = (dongle_id or "").strip().lower() or None
  probed_info: dict[str, Any] | None = None
  if detected_id is None:
    probed = probe_carrot_device(clean_host, port=parsed_port, connect_timeout_s=1.0, http_timeout_s=min(3.0, resolved_timeout))
    if probed is not None:
      _, probed_info = probed
      detected_id = extract_dongle_id(probed_info) or None
  saved = save_config(
    host=clean_host,
    port=parsed_port,
    timeout_s=resolved_timeout,
    dongle_id=detected_id,
    config_path=config_path,
  )
  return {"config": saved, "probed": probed_info is not None, "info": probed_info}


def _normalize_host_and_port(host: str, port: int = DEFAULT_PORT) -> tuple[str, int]:
  raw = host.strip()
  if "://" in raw:
    parsed = urlparse(raw)
    clean_host = parsed.hostname or raw
    resolved_port = parsed.port or port
    return clean_host, int(resolved_port)
  if ":" in raw and not raw.startswith("["):
    h, maybe_port = raw.rsplit(":", 1)
    if maybe_port.isdigit():
      return h.strip(), int(maybe_port)
  return raw.strip("/"), int(port)


def local_ipv4_addresses() -> list[str]:
  ips: list[str] = []
  seen: set[str] = set()

  def _add(addr: str) -> None:
    try:
      ip = ipaddress.IPv4Address(addr)
    except ValueError:
      return
    if ip.is_loopback or ip.is_link_local or ip.is_multicast or ip.is_unspecified:
      return
    if ip.is_private and addr not in seen:
      seen.add(addr)
      ips.append(addr)

  for probe_target in (("1.1.1.1", 80), ("192.168.1.1", 80), ("10.0.0.1", 80)):
    try:
      with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        s.connect(probe_target)
        _add(s.getsockname()[0])
    except OSError:
      continue

  try:
    for info in socket.getaddrinfo(socket.gethostname(), None, family=socket.AF_INET):
      _add(info[4][0])
  except OSError:
    pass

  return ips


def arp_neighbor_ips(allowed_networks: Sequence[ipaddress.IPv4Network]) -> list[str]:
  if not allowed_networks:
    return []
  try:
    out = subprocess.check_output(["arp", "-an"], text=True, timeout=2.0, stderr=subprocess.DEVNULL)
  except Exception:
    return []
  neighbors: list[str] = []
  seen: set[str] = set()
  for match in re.findall(r"\((\d+\.\d+\.\d+\.\d+)\)", out):
    try:
      ip = ipaddress.IPv4Address(match)
    except ValueError:
      continue
    if ip.is_loopback or ip.is_multicast or match.endswith(".255") or match.endswith(".0"):
      continue
    if any(ip in net for net in allowed_networks) and match not in seen:
      seen.add(match)
      neighbors.append(match)
  return neighbors


def probe_carrot_device(
  host: str,
  port: int = DEFAULT_PORT,
  connect_timeout_s: float = 0.45,
  http_timeout_s: float = 2.0,
) -> tuple[str, dict[str, Any]] | None:
  try:
    with socket.create_connection((host, int(port)), timeout=connect_timeout_s):
      pass
  except OSError:
    return None
  try:
    info = CarrotDeviceClient(host, port=int(port), timeout_s=http_timeout_s).get_device_info()
    return host, info
  except Exception:
    return None


def discover_device_on_wifi(
  *,
  dongle_id: str | None = None,
  port: int = DEFAULT_PORT,
  subnets: Sequence[str] | None = None,
  candidate_ips: Sequence[str] | None = None,
  probe_timeout_s: float = 0.45,
  max_workers: int = 48,
) -> tuple[str, dict[str, Any]] | None:
  target_id = dongle_id.strip().lower() if dongle_id else None

  if candidate_ips is not None:
    targets = list(dict.fromkeys(str(ip).strip() for ip in candidate_ips if str(ip).strip()))
  else:
    self_ips = set(local_ipv4_addresses())
    if subnets:
      networks = [ipaddress.IPv4Network(s, strict=False) for s in subnets]
    else:
      networks = []
      seen_net: set[str] = set()
      for local_ip in self_ips:
        net = ipaddress.IPv4Network(f"{local_ip}/24", strict=False)
        if str(net) not in seen_net:
          seen_net.add(str(net))
          networks.append(net)

    if not networks:
      return None

    ordered: list[str] = []
    seen_target: set[str] = set()
    for ip_str in arp_neighbor_ips(networks):
      if ip_str not in self_ips and ip_str not in seen_target:
        seen_target.add(ip_str)
        ordered.append(ip_str)
    for net in networks:
      for host_ip in net.hosts():
        ip_str = str(host_ip)
        if ip_str not in self_ips and ip_str not in seen_target:
          seen_target.add(ip_str)
          ordered.append(ip_str)
    targets = ordered

  if not targets:
    return None

  matches: list[tuple[str, dict[str, Any]]] = []
  pool_size = max(1, min(int(max_workers), len(targets)))
  with ThreadPoolExecutor(max_workers=pool_size) as pool:
    futs = {
      pool.submit(probe_carrot_device, host_ip, port, probe_timeout_s, 2.0): host_ip
      for host_ip in targets
    }
    for fut in as_completed(futs):
      res = fut.result()
      if res is None:
        continue
      found_host, info = res
      found_id = extract_dongle_id(info)
      if target_id is not None:
        if found_id == target_id:
          for pending in futs:
            pending.cancel()
          return found_host, info
      else:
        matches.append((found_host, info))

  if target_id is None:
    if len(matches) == 1:
      return matches[0]
    if len(matches) > 1:
      found_list = ", ".join(f"{h}({extract_dongle_id(i) or 'unknown'})" for h, i in matches)
      raise RuntimeError(
        f"같은 Wi-Fi에 여러 Carrot 단말이 발견되었습니다: {found_list}. "
        "`--dongle-id <ID>`를 지정해 주세요."
      )
  return None


def resolve_device_target(
  host: str | None = None,
  port: int | None = None,
  timeout_s: float | None = None,
  dongle_id: str | None = None,
  auto_discover: bool = True,
  config_path: Path = CONFIG_PATH,
) -> tuple[str, int, float]:
  cfg = load_config(config_path)
  resolved_port = int(port or cfg.get("port") or DEFAULT_PORT)
  resolved_timeout = float(timeout_s or cfg.get("timeout_s") or DEFAULT_TIMEOUT_S)
  expected_id = (
    (dongle_id or os.environ.get("COMMA_DONGLE_ID") or cfg.get("dongle_id") or "").strip().lower() or None
  )

  explicit = host or os.environ.get("COMMA_HOST")
  if explicit:
    clean_host, explicit_port = _normalize_host_and_port(str(explicit), resolved_port)
    if expected_id is None and auto_discover:
      probed = probe_carrot_device(clean_host, port=explicit_port, connect_timeout_s=1.0, http_timeout_s=min(3.0, resolved_timeout))
      if probed is not None:
        _, info = probed
        found_id = extract_dongle_id(info)
        if found_id:
          save_config(host=clean_host, port=explicit_port, timeout_s=resolved_timeout, dongle_id=found_id, config_path=config_path)
    return clean_host, explicit_port, resolved_timeout

  saved_host = cfg.get("host")
  if saved_host:
    clean_saved, saved_port = _normalize_host_and_port(str(saved_host), resolved_port)
    if not auto_discover:
      return clean_saved, saved_port, resolved_timeout
    probed = probe_carrot_device(clean_saved, port=saved_port, connect_timeout_s=1.0, http_timeout_s=min(3.0, resolved_timeout))
    if probed is not None:
      _, info = probed
      found_id = extract_dongle_id(info)
      if expected_id is None or found_id == expected_id:
        if found_id and not cfg.get("dongle_id"):
          save_config(host=clean_saved, port=saved_port, timeout_s=resolved_timeout, dongle_id=found_id, config_path=config_path)
        return clean_saved, saved_port, resolved_timeout

  if auto_discover:
    discovered = discover_device_on_wifi(dongle_id=expected_id, port=resolved_port)
    if discovered is not None:
      found_host, info = discovered
      found_id = extract_dongle_id(info) or expected_id
      save_config(host=found_host, port=resolved_port, timeout_s=resolved_timeout, dongle_id=found_id, config_path=config_path)
      return found_host, resolved_port, resolved_timeout

  for candidate in DEFAULT_CANDIDATE_HOSTS:
    probed = probe_carrot_device(candidate, port=resolved_port, connect_timeout_s=1.0, http_timeout_s=min(3.0, resolved_timeout))
    if probed is None:
      continue
    _, info = probed
    found_id = extract_dongle_id(info)
    if expected_id is None or found_id == expected_id:
      save_config(host=candidate, port=resolved_port, timeout_s=resolved_timeout, dongle_id=found_id or expected_id, config_path=config_path)
      return candidate, resolved_port, resolved_timeout

  id_hint = f" (dongle_id={expected_id})" if expected_id else ""
  raise RuntimeError(
    f"동일 Wi-Fi에서 오파 단말{id_hint}을 찾지 못했습니다. "
    "단말과 맥북이 같은 Wi-Fi에 있는지 확인하거나 `config --host <IP>` / `config --dongle-id <ID>`를 설정해 주세요."
  )


def _infer_rlog_filename(segment: str, content_disposition: str | None, content_type: str | None) -> str:
  safe_seg = _SAFE_NAME_RE.sub("_", segment)
  if content_disposition:
    m = _FILENAME_RE.search(content_disposition)
    if m:
      fname = Path(m.group(1)).name
      if fname.endswith(".zst"):
        return f"{safe_seg}_rlog.zst"
      if fname.endswith(".bz2"):
        return f"{safe_seg}_rlog.bz2"
      if fname.endswith("rlog"):
        return f"{safe_seg}_rlog"
  ctype = (content_type or "").lower()
  if "bzip2" in ctype:
    return f"{safe_seg}_rlog.bz2"
  if "zstd" in ctype or "zst" in ctype:
    return f"{safe_seg}_rlog.zst"
  return f"{safe_seg}_rlog.zst"


def _find_existing_rlog(dest_dir: Path, segment: str) -> Path | None:
  safe_seg = _SAFE_NAME_RE.sub("_", segment)
  for candidate_name in (
    f"{safe_seg}_rlog.zst",
    f"{safe_seg}_rlog.bz2",
    f"{safe_seg}_rlog",
    f"{segment}--rlog.zst",
    f"{segment}--rlog.bz2",
  ):
    p = dest_dir / candidate_name
    if p.is_file() and p.stat().st_size > 0:
      return p
  return None


class CarrotDeviceClient:
  """Minimal HTTP client for Carrot Web (port 7000) endpoints."""

  def __init__(self, host: str, port: int = DEFAULT_PORT, timeout_s: float = DEFAULT_TIMEOUT_S):
    clean_host, resolved_port = _normalize_host_and_port(host, port)
    self.host = clean_host
    self.port = resolved_port
    self.timeout_s = float(timeout_s)
    self.base_url = f"http://{self.host}:{self.port}"

  def _get_json(self, path: str) -> dict[str, Any]:
    req = Request(f"{self.base_url}{path}", headers={"Accept": "application/json"})
    try:
      with urlopen(req, timeout=self.timeout_s) as resp:
        raw = resp.read().decode("utf-8")
        data = json.loads(raw)
        if not isinstance(data, dict):
          raise ValueError(f"unexpected JSON payload from {path}")
        return data
    except HTTPError as e:
      raise RuntimeError(f"HTTP {e.code} from {self.base_url}{path}: {e.reason}") from e
    except URLError as e:
      raise RuntimeError(f"연결 실패 ({self.base_url}{path}): {e.reason}") from e

  def get_device_info(self) -> dict[str, Any]:
    data = self._get_json("/api/tools/device_info")
    info = data.get("info")
    return info if isinstance(info, dict) else data

  def list_drive_reports(self, limit: int = 50) -> list[dict[str, Any]]:
    data = self._get_json(f"/api/drive-reports?limit={max(1, min(100, int(limit)))}")
    reports = data.get("reports")
    return list(reports) if isinstance(reports, list) else []

  def list_routes(self, limit: int = 20, segment_limit: int = 2000) -> list[dict[str, Any]]:
    data = self._get_json(
      f"/api/dashcam/routes?offset=0&limit={max(1, int(limit))}&segment_limit={max(1, int(segment_limit))}&sort=asc"
    )
    routes = data.get("routes")
    return list(routes) if isinstance(routes, list) else []

  def list_segments(self, route: str) -> list[str]:
    encoded = quote(route, safe="")
    segments: list[str] = []
    offset = 0
    while True:
      data = self._get_json(f"/api/dashcam/segments/{encoded}?offset={offset}&limit=2000&sort=asc")
      batch = data.get("segments") or []
      segments.extend(str(s) for s in batch)
      if not data.get("hasMore") or data.get("nextOffset") is None:
        break
      offset = int(data["nextOffset"])
    return segments

  def download_rlog(self, segment: str, dest_dir: Path, chunk_size: int = 65536) -> tuple[Path, bool, int]:
    dest_dir.mkdir(parents=True, exist_ok=True)
    existing = _find_existing_rlog(dest_dir, segment)
    if existing is not None:
      return existing, False, 0

    encoded = quote(segment, safe="")
    endpoints = (
      f"/api/dashcam/download/{encoded}/rlog",
      f"/api/dashcam/replay-source/{encoded}/rlog",
    )
    last_err: Exception | None = None
    for endpoint in endpoints:
      url = f"{self.base_url}{endpoint}"
      req = Request(url)
      try:
        with urlopen(req, timeout=max(30.0, self.timeout_s * 4.0)) as resp:
          fname = _infer_rlog_filename(
            segment,
            resp.headers.get("Content-Disposition"),
            resp.headers.get("Content-Type"),
          )
          target_path = dest_dir / fname
          part_path = dest_dir / f".{fname}.part"
          total = 0
          try:
            with open(part_path, "wb") as fh:
              while True:
                chunk = resp.read(chunk_size)
                if not chunk:
                  break
                fh.write(chunk)
                total += len(chunk)
            if total == 0:
              raise RuntimeError(f"empty rlog stream for {segment}")
            os.replace(part_path, target_path)
            return target_path, True, total
          except Exception:
            if part_path.exists():
              part_path.unlink(missing_ok=True)
            raise
      except HTTPError as e:
        last_err = e
        if e.code == 404:
          continue
        raise RuntimeError(f"HTTP {e.code} downloading {segment}: {e.reason}") from e
      except URLError as e:
        last_err = e
        raise RuntimeError(f"다운로드 연결 오류 ({segment}): {e.reason}") from e

    raise RuntimeError(f"rlog 파일을 찾을 수 없습니다 ({segment}): {last_err}")


def _sync_reports(client: CarrotDeviceClient, reports_dir: Path, limit: int = 50) -> tuple[int, int]:
  reports = client.list_drive_reports(limit=limit)
  saved = 0
  skipped = 0
  for idx, report in enumerate(reports):
    if not isinstance(report, dict):
      continue
    report_id = str(report.get("report_id") or f"report_{idx}")
    safe_id = _SAFE_NAME_RE.sub("_", report_id)
    out_path = reports_dir / f"{safe_id}.json"
    encoded = json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    if out_path.is_file() and out_path.read_text(encoding="utf-8") == encoded:
      skipped += 1
    else:
      out_path.write_text(encoded, encoding="utf-8")
      saved += 1
  return saved, skipped


def sync_from_device(
  host: str | None = None,
  port: int | None = None,
  timeout_s: float | None = None,
  dongle_id: str | None = None,
  latest_routes: int = 3,
  route_filter: str | None = None,
  max_segments_per_route: int | None = None,
  workers: int = 3,
  base_dir: Path | None = None,
  config_path: Path = CONFIG_PATH,
) -> SyncResult:
  dirs = ensure_analysis_dirs(base_dir)
  resolved_host, resolved_port, resolved_timeout = resolve_device_target(
    host=host,
    port=port,
    timeout_s=timeout_s,
    dongle_id=dongle_id,
    config_path=config_path,
  )
  client = CarrotDeviceClient(resolved_host, port=resolved_port, timeout_s=resolved_timeout)
  result = SyncResult(host=resolved_host, port=resolved_port)

  try:
    result.device_info = client.get_device_info()
  except Exception as e:
    result.errors.append(f"device_info: {e}")

  try:
    saved, skipped = _sync_reports(client, dirs["reports"], limit=50)
    result.reports_saved = saved
    result.reports_skipped = skipped
  except Exception as e:
    result.errors.append(f"drive_reports: {e}")

  routes = client.list_routes(limit=max(50 if route_filter else 20, int(latest_routes)), segment_limit=2000)
  if route_filter:
    exact = [r for r in routes if str(r.get("route") or "").strip() == route_filter]
    selected = exact if exact else [r for r in routes if route_filter in str(r.get("route") or "")]
  else:
    selected = routes[: max(1, int(latest_routes))]

  tasks: list[tuple[str, str, Path]] = []
  for entry in selected:
    rname = str(entry.get("route") or "").strip()
    if not rname:
      continue
    safe_route = _SAFE_NAME_RE.sub("_", rname)
    route_dir = dirs["routes"] / safe_route
    route_dir.mkdir(parents=True, exist_ok=True)
    result.routes_synced.append(rname)

    if entry.get("segmentsHasMore"):
      segments = client.list_segments(rname)
    else:
      segments = [str(s) for s in (entry.get("segmentFolders") or [])]

    if max_segments_per_route is not None and max_segments_per_route > 0:
      segments = segments[-int(max_segments_per_route) :]

    for seg in segments:
      tasks.append((rname, seg, route_dir))

  if tasks:
    pool_workers = max(1, min(int(workers), len(tasks)))
    with ThreadPoolExecutor(max_workers=pool_workers) as pool:
      futures = {
        pool.submit(client.download_rlog, seg, route_dir): (rname, seg)
        for rname, seg, route_dir in tasks
      }
      for fut in as_completed(futures):
        _rname, seg = futures[fut]
        try:
          path, downloaded, nbytes = fut.result()
          if downloaded:
            result.segments_downloaded += 1
            result.bytes_downloaded += nbytes
          else:
            result.segments_skipped += 1
          result.downloaded_files.append(str(path))
        except Exception as e:
          result.errors.append(f"{seg}: {e}")

  result.downloaded_files.sort()
  return result


def add_to_golden(paths: list[str | Path], golden_dir: Path = GOLDEN_DIR) -> tuple[list[Path], list[Path]]:
  """Copy verified baseline rlogs into golden_v1.0/ without overwriting existing identical files."""
  golden_dir.mkdir(parents=True, exist_ok=True)
  added: list[Path] = []
  skipped: list[Path] = []
  sources: list[Path] = []

  for item in paths:
    p = Path(item).expanduser().resolve()
    if p.is_file() and (p.name.endswith(".zst") or p.name.endswith(".bz2")):
      sources.append(p)
    elif p.is_dir():
      for child in sorted(p.rglob("*")):
        if child.is_file() and (child.name.endswith("rlog.zst") or child.name.endswith("rlog.bz2")):
          sources.append(child)

  for src in sources:
    if src.name in ("rlog.zst", "rlog.bz2") and src.parent.name:
      dst_name = f"{src.parent.name}_{src.name}"
    else:
      dst_name = src.name
    dst = golden_dir / dst_name
    if dst.is_file() and dst.stat().st_size == src.stat().st_size:
      skipped.append(dst)
      continue
    shutil.copy2(src, dst)
    added.append(dst)

  return added, skipped


def _resolve_route_times(entry: dict[str, Any]) -> tuple[float, float, float]:
  """Return (start_epoch, end_epoch, duration_min), guarding against pre-NTP boot clock jumps."""
  seg_times = entry.get("segmentTimes")
  seg_folders = [str(s) for s in (entry.get("segmentFolders") or [])]
  ordered_pairs: list[tuple[float, float]] = []
  if isinstance(seg_times, dict) and seg_times:
    keys = seg_folders if seg_folders else sorted(
      seg_times.keys(),
      key=lambda k: int(k.rsplit("--", 1)[-1]) if "--" in str(k) and str(k).rsplit("--", 1)[-1].isdigit() else 0,
    )
    for k in keys:
      item = seg_times.get(k)
      if isinstance(item, dict):
        s_ep = float(item.get("startEpoch") or 0.0)
        e_ep = float(item.get("endEpoch") or 0.0)
        if e_ep > 1e9:
          ordered_pairs.append((s_ep if s_ep > 1e9 else e_ep, e_ep))

  end_ep = float(entry.get("routeEndEpoch") or entry.get("latestModifiedEpoch") or 0.0)
  if end_ep <= 1e9 and ordered_pairs:
    end_ep = max(p[1] for p in ordered_pairs)
  start_ep = float(entry.get("routeStartEpoch") or 0.0)
  if start_ep <= 1e9 and ordered_pairs:
    start_ep = ordered_pairs[0][0]
  if end_ep <= 1e9:
    return 0.0, 0.0, 0.0

  seg_count = int(entry.get("segmentCount") or len(seg_folders) or len(ordered_pairs) or 1)
  max_plausible_span = max(3600.0, seg_count * 180.0)
  if start_ep <= 1e9 or end_ep < start_ep or (end_ep - start_ep) > max_plausible_span:
    if ordered_pairs:
      tail_start = ordered_pairs[-1][0]
      prev_s, prev_e = ordered_pairs[-1]
      if prev_e - prev_s > 600.0:
        tail_start = max(prev_s, prev_e - 60.0)
      else:
        for s_ep, e_ep in reversed(ordered_pairs[:-1]):
          if (e_ep - s_ep) > 600.0 or (prev_s - e_ep) > 300.0 or s_ep > prev_s:
            break
          tail_start = s_ep
          prev_s = s_ep
      start_ep = tail_start
    else:
      start_ep = end_ep - seg_count * 60.0

  dur_min = round(max(0.0, end_ep - start_ep) / 60.0, 1)
  return start_ep, end_ep, dur_min


def _load_local_reports_raw(reports_dir: Path) -> list[dict[str, Any]]:
  if not reports_dir.is_dir():
    return []
  out: list[dict[str, Any]] = []
  for p in sorted(reports_dir.glob("*.json"), reverse=True):
    if p.name == "index.json":
      continue
    try:
      data = json.loads(p.read_text(encoding="utf-8"))
      if isinstance(data, dict) and "report_id" in data:
        out.append(data)
    except Exception:
      continue
  return out


def _match_drive_report(
  start_epoch: float,
  end_epoch: float,
  reports: Sequence[dict[str, Any]],
  *,
  tolerance_s: float = 360.0,
) -> dict[str, Any] | None:
  if end_epoch <= 1e9 or not reports:
    return None
  best: dict[str, Any] | None = None
  best_diff = tolerance_s + 1.0
  for r in reports:
    r_end = float(r.get("ended_at") or 0.0)
    r_start = float(r.get("started_at") or 0.0)
    if r_end <= 1e9:
      continue
    diff = abs(r_end - end_epoch)
    overlaps = start_epoch > 1e9 and r_start > 1e9 and (r_start <= end_epoch + 180.0 and r_end >= start_epoch - 180.0)
    if (diff <= tolerance_s or overlaps) and diff < best_diff:
      best_diff = diff
      best = {
        "report_id": r.get("report_id"),
        "score": r.get("score"),
        "grade": r.get("grade"),
        "distance_km": r.get("distance_km"),
        "auto_hard_brake_count": r.get("auto_hard_brake_count", r.get("hard_brake_count", 0)),
        "manual_hard_brake_count": r.get("manual_hard_brake_count", 0),
      }
  return best


def _local_rlogs_in_dir(route_dir: Path) -> list[Path]:
  if not route_dir.is_dir():
    return []
  return [
    p for p in route_dir.rglob("*")
    if p.is_file() and (p.name.endswith("_rlog.zst") or p.name.endswith("_rlog.bz2") or p.name in ("rlog.zst", "rlog.bz2") or p.name.endswith("--rlog.zst"))
  ]


def _format_epoch(ts: float, fmt: str = "%Y-%m-%d %H:%M") -> str:
  if ts <= 1e9:
    return ""
  return datetime.fromtimestamp(ts).strftime(fmt)


def list_route_groups(
  *,
  host: str | None = None,
  port: int | None = None,
  timeout_s: float | None = None,
  dongle_id: str | None = None,
  limit: int = 15,
  local_only: bool = False,
  base_dir: Path | None = None,
  config_path: Path = CONFIG_PATH,
) -> tuple[list[RouteGroup], str, str]:
  """List multi-segment route groups across live Carrot device and local cache (newest first)."""
  dirs = ensure_analysis_dirs(base_dir)
  routes_root = dirs["routes"]
  reports_root = dirs["reports"]
  scans_root = dirs["scans"]

  device_routes: list[dict[str, Any]] = []
  device_label = ""
  device_error = ""

  if not local_only:
    try:
      resolved_host, resolved_port, resolved_timeout = resolve_device_target(
        host=host,
        port=port,
        timeout_s=timeout_s,
        dongle_id=dongle_id,
        config_path=config_path,
      )
      client = CarrotDeviceClient(resolved_host, port=resolved_port, timeout_s=resolved_timeout)
      device_label = f"{resolved_host}:{resolved_port}"
      try:
        _sync_reports(client, reports_root, limit=50)
      except Exception:
        pass
      device_routes = client.list_routes(limit=max(20, int(limit)), segment_limit=2000)
    except Exception as e:
      device_error = str(e)

  reports = _load_local_reports_raw(reports_root)
  merged: dict[str, RouteGroup] = {}

  # Scan local route directories first
  if routes_root.is_dir():
    for d in sorted(routes_root.iterdir()):
      if not d.is_dir():
        continue
      rname = d.name
      rlogs = _local_rlogs_in_dir(d)
      if not rlogs:
        continue
      safe_route = _SAFE_NAME_RE.sub("_", rname)
      scan_dir = scans_root / f"route_{safe_route}"
      html_path = scan_dir / "analysis_report.html"
      summary_path = scan_dir / "scan_summary.json"
      start_ep = 0.0
      end_ep = 0.0
      dur_min = 0.0
      if summary_path.is_file():
        try:
          sd = json.loads(summary_path.read_text(encoding="utf-8"))
          segs = [s for s in (sd.get("segments") or []) if not s.get("error")]
          ts_list = [float(s.get("started_at_ts") or 0.0) for s in segs if float(s.get("started_at_ts") or 0.0) > 1e9]
          if ts_list:
            start_ep = min(ts_list)
            last_seg = max(segs, key=lambda s: float(s.get("started_at_ts") or 0.0))
            end_ep = float(last_seg.get("started_at_ts") or start_ep) + float(last_seg.get("duration_s") or 0.0)
            dur_min = round(sum(float(s.get("duration_s") or 0.0) for s in segs) / 60.0, 1)
        except Exception:
          pass
      if end_ep <= 1e9:
        mtimes = [p.stat().st_mtime for p in rlogs]
        end_ep = max(mtimes)
        start_ep = min(mtimes)
      merged[rname] = RouteGroup(
        route=rname,
        source="local",
        device_segments=0,
        local_segments=len(rlogs),
        started_at=_format_epoch(start_ep),
        ended_at=_format_epoch(end_ep, "%H:%M"),
        start_epoch=start_ep,
        end_epoch=end_ep,
        duration_min=dur_min,
        local_dir=str(d),
        matched_report=_match_drive_report(start_ep, end_ep, reports),
        analyzed_html=str(html_path) if html_path.is_file() else "",
      )

  for entry in device_routes:
    rname = str(entry.get("route") or "").strip()
    if not rname:
      continue
    safe_route = _SAFE_NAME_RE.sub("_", rname)
    local_dir_path = routes_root / safe_route
    rlogs = _local_rlogs_in_dir(local_dir_path)
    dev_count = int(entry.get("segmentCount") or len(entry.get("segmentFolders") or []) or 0)
    start_ep, end_ep, dur_min = _resolve_route_times(entry)
    scan_dir = scans_root / f"route_{safe_route}"
    html_path = scan_dir / "analysis_report.html"
    matched = _match_drive_report(start_ep, end_ep, reports)
    existing = merged.get(rname)
    merged[rname] = RouteGroup(
      route=rname,
      source="device+local" if rlogs else "device",
      device_segments=dev_count,
      local_segments=len(rlogs) if rlogs else (existing.local_segments if existing else 0),
      started_at=_format_epoch(start_ep) or (existing.started_at if existing else ""),
      ended_at=_format_epoch(end_ep, "%H:%M") or (existing.ended_at if existing else ""),
      start_epoch=start_ep or (existing.start_epoch if existing else 0.0),
      end_epoch=end_ep or (existing.end_epoch if existing else 0.0),
      duration_min=dur_min or (existing.duration_min if existing else 0.0),
      latest_label=str(entry.get("latestModifiedLabel") or ""),
      local_dir=str(local_dir_path) if rlogs else "",
      matched_report=matched or (existing.matched_report if existing else None),
      analyzed_html=str(html_path) if html_path.is_file() else (existing.analyzed_html if existing else ""),
    )

  groups = sorted(merged.values(), key=lambda g: (g.end_epoch, g.route), reverse=True)
  if limit > 0:
    groups = groups[: int(limit)]
  return groups, device_label, device_error


def render_route_groups_markdown(
  groups: Sequence[RouteGroup],
  *,
  device_label: str = "",
  device_error: str = "",
) -> str:
  lines: list[str] = ["# 최근 주행로그 그룹 목록 (루트 단위)", ""]
  if device_label:
    lines.append(f"- 연결 단말: `{device_label}`")
  elif device_error:
    lines.append(f"- 단말 미연결 (로컬 캐시 기준 표시): {device_error}")
  lines.append("")
  if not groups:
    lines.append("- 표시할 주행로그 그룹이 없습니다.")
    return "\n".join(lines)

  lines.append("| # | 주행로그 그룹 | 주행일시 | 시간 | 세그먼트(로컬/단말) | 주행리포트 | 상태 |")
  lines.append("|---:|---|---|---:|---:|---|---|")
  for idx, g in enumerate(groups, 1):
    dt = f"{g.started_at}~{g.ended_at}" if (g.started_at and g.ended_at) else (g.started_at or "-")
    if g.latest_label:
      dt = f"{dt} ({g.latest_label})"
    dur = f"{g.duration_min:.0f}분" if g.duration_min > 0 else "-"
    seg_col = f"{g.local_segments} / {g.device_segments if g.device_segments else '-'}"
    rep = g.matched_report
    rep_col = f"{rep.get('score')}점({rep.get('grade')}) {float(rep.get('distance_km', 0) or 0):.1f}km" if rep else "-"
    if g.analyzed_html:
      st = "분석완료"
    elif g.local_segments >= g.device_segments and g.local_segments > 0:
      st = "로컬동기화됨"
    elif g.local_segments > 0:
      st = "일부동기화"
    else:
      st = "단말보관"
    lines.append(f"| {idx} | `{g.route}` | {dt} | {dur} | {seg_col} | {rep_col} | {st} |")
  lines.append("")
  lines.append("- 선택한 그룹 분석: `PYTHONPATH=. uv run --no-sync python3 -m openpilot.tools.drive_analyzer analyze --route <번호 또는 루트명>`")
  return "\n".join(lines)
