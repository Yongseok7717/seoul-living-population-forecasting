"""Paths and constants shared by the data pipeline, the models and the experiments.

Data root: set the environment variable LIVPOP_DATA to the folder that holds the raw downloads and receives processed/
(default: <repo>/data).  Expected layout:
    <LIVPOP_DATA>/250m격자 단위/                Seoul Living Population, 250 m grid (Seoul Open Data Plaza) and the grid shapefiles (match/)
    <LIVPOP_DATA>/[서울열린데이터 광장]/        facility, hazard (rainfall) and transit downloads (data_pipeline/fetch_*.py)
    <LIVPOP_DATA>/data/                          compiled tables: venue schedule, emergency-alert messages (data_pipeline/build_venue_events.py,
                                                 data_pipeline/fetch_disaster_msgs.py)
    <LIVPOP_DATA>/processed/                     everything the data pipeline builds
Model outputs go to <repo>/results (base runs in results/base/{results,ckpt,logs}, one folder per experiment otherwise).
API keys, needed only by the fetch scripts, are read from KEY=VALUE lines in <repo>/.env (see .env.example).
"""
import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
_ENV = REPO / ".env"
if _ENV.exists():
    for _line in _ENV.read_text(encoding="utf-8").splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _k, _v = _line.split("=", 1); os.environ.setdefault(_k.strip(), _v.strip().strip('"').strip("'"))

ROOT = Path(os.environ.get("LIVPOP_DATA", REPO / "data")).resolve()
GRID_DIR = ROOT / "250m격자 단위"
RAW_LOCAL = GRID_DIR / "250M격자_생활인구(내국인)"
RAW_FORN_LONG = GRID_DIR / "250M격자_생활인구(장기외국인)"
RAW_FORN_TEMP = GRID_DIR / "250M격자_생활인구(단기외국인)"
MATCH_SHP = GRID_DIR / "match" / "match.shp"            # EPSG:5179, 10,125 cells
DONG_CODE_CSV = GRID_DIR / "match" / "250_LOCAL_RESD_행정동코드.csv"
OPEN_DIR = ROOT / "[서울열린데이터 광장]"                # downloads of the fetch scripts
TABLES = ROOT / "data"                                   # compiled tables (venue schedule, emergency alerts)

RESULTS = REPO / "results"
BASE = RESULTS / "base"                                  # base runs: results/base/results/<name>.json, results/base/ckpt/<name>.pt

OUT = ROOT / "processed"
OUT_GRID = OUT / "grid"
OUT_LOCAL = OUT / "tensors" / "local"
OUT_FORN = OUT / "tensors" / "foreign"
OUT_QC = OUT / "qc"

CRS_STANDARD = 5179   # Korea 2000 / Unified CS (match.shp native)
CELL_SIZE = 250.0

# Months covered by the 250m domestic product (inclusive)
LOCAL_MONTHS = [f"{y}{m:02d}" for y in (2023, 2024, 2025) for m in range(1, 13)] + ["202601"]
FORN_MONTHS = [f"{y}{m:02d}" for y in (2023, 2024, 2025) for m in range(1, 13)]

# Domestic file schema (33 columns). Age bands: 00=0-9, 10=10-14, ..., 65=65-69, 70=70+
AGE_BANDS = ["00", "10", "15", "20", "25", "30", "35", "40", "45", "50", "55", "60", "65", "70"]
GROUP_COLS = [f"M{a}" for a in AGE_BANDS] + [f"F{a}" for a in AGE_BANDS]   # 28 columns
LOCAL_VALUE_COLS = ["SPOP"] + GROUP_COLS                                   # 29 columns stored
# 0-9 is a derived column (manual step 4) and under-represented; analysis total excludes it.
EXCLUDED_GROUPS = ["M00", "F00"]
POP10_COLS = [c for c in GROUP_COLS if c not in EXCLUDED_GROUPS]           # 26 columns

# Foreigner nationality codes (file order is identical in the English and Korean headers)
FORN_LONG_CODES = ["NPL", "RUS", "MNG", "USA", "MMR", "BGD", "VNM", "LKA", "UZB", "IND", "IDN",
                   "JPN", "CHN", "KAZ", "KAM", "CAN", "THA", "PAK", "FRA", "PHL", "ETC"]
FORN_TEMP_CODES = [c for c in FORN_LONG_CODES if c not in ("MMR", "BGD")]
FORN_KO_TO_CODE = {"네팔": "NPL", "러시아": "RUS", "몽골": "MNG", "미국": "USA", "미얀마": "MMR",
                   "방글라데시": "BGD", "베트남": "VNM", "스리랑카": "LKA", "우즈베키스탄": "UZB",
                   "인도": "IND", "인도네시아": "IDN", "일본": "JPN", "중국": "CHN", "카자흐스탄": "KAZ",
                   "캄보디아": "KAM", "캐나다": "CAN", "태국": "THA", "파키스탄": "PAK", "프랑스": "FRA",
                   "필리핀": "PHL", "기타": "ETC"}


def ensure_dirs():
    for d in (OUT_GRID, OUT_LOCAL, OUT_FORN, OUT_QC):
        d.mkdir(parents=True, exist_ok=True)
