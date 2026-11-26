"""MacBook-side Wi-Fi log sync, 1st-pass anomaly scanner, and ver-1.0 regression gate."""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
ANALYSIS_DIR = REPO_ROOT / ".analysis" / "drive_analyzer"
CONFIG_PATH = ANALYSIS_DIR / "config.json"
REPORTS_DIR = ANALYSIS_DIR / "reports"
ROUTES_DIR = ANALYSIS_DIR / "routes"
SCANS_DIR = ANALYSIS_DIR / "scans"
GOLDEN_DIR = ANALYSIS_DIR / "golden_v1.0"

DEFAULT_PORT = 7000
DEFAULT_TIMEOUT_S = 15.0
DEFAULT_BASELINE_REF = "ver-1.0"
DEFAULT_CANDIDATE_HOSTS = ("carrot.local", "comma.local")


def ensure_analysis_dirs(base_dir: Path | None = None) -> dict[str, Path]:
  """Create local gitignored directories for synced reports, routes, scans, and golden logs."""
  root = Path(base_dir) if base_dir is not None else ANALYSIS_DIR
  dirs = {
    "root": root,
    "reports": root / "reports",
    "routes": root / "routes",
    "scans": root / "scans",
    "golden": root / "golden_v1.0",
  }
  for d in dirs.values():
    d.mkdir(parents=True, exist_ok=True)
  return dirs
