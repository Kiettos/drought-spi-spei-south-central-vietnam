"""
src/data/download.py
====================
Chỉ lo phần TẢI dữ liệu thô (raw). Không xử lý, không cắt vùng.

Chức năng:
    - Tải ERA5 monthly (tp, t2m) qua Copernicus CDS API.
    - Tải CHIRPS monthly (mưa vệ tinh) từ UCSB.
    - Tải SPI/SPEI có sẵn từ ERA5-Drought (tùy chọn).
    - Bỏ qua file đã có (idempotent), có --overwrite.

Cách dùng:
    python -m src.data.download era5
    python -m src.data.download chirps
    python -m src.data.download spi-spei
    python -m src.data.download all --overwrite

Yêu cầu:
    - ERA5: cấu hình ~/.cdsapirc (xem README).
    - CHIRPS, ERA5-Drought: không cần API key.
"""

from __future__ import annotations

import argparse
import gzip
import logging
import shutil
import sys
import urllib.request
from pathlib import Path
from typing import Optional

import pandas as pd
import yaml

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
def load_config(path: str | Path = "config/config.yaml") -> dict:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    for k, v in cfg.get("paths", {}).items():
        cfg["paths"][k] = Path(v)
        cfg["paths"][k].mkdir(parents=True, exist_ok=True)
    return cfg


# ---------------------------------------------------------------------------
# ERA5 monthly — Copernicus CDS
# ---------------------------------------------------------------------------
ERA5_VAR_MAP = {
    "tp": "total_precipitation",
    "total_precipitation": "total_precipitation",
    "t2m": "2m_temperature",
    "2m_temperature": "2m_temperature",
}


def download_era5_monthly(
    cfg: dict,
    overwrite: bool = False,
    variables: Optional[list[str]] = None,
) -> Path:
    """
    Tải ERA5 monthly means cho bbox + giai đoạn trong config.

    Trả về đường dẫn file .nc đã lưu tại data/raw/era5/.
    """
    try:
        import cdsapi
    except ImportError:
        raise ImportError(
            "Chưa cài cdsapi. Chạy `pip install cdsapi` và tạo ~/.cdsapirc."
        )

    region = cfg["region"]
    lon_min, lat_min, lon_max, lat_max = region["bbox"]
    start = pd.Timestamp(cfg["time"]["start"])
    end = pd.Timestamp(cfg["time"]["end"])

    out_dir = cfg["paths"]["raw"] / "era5"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"era5_monthly_{region['name']}_{start:%Y%m}_{end:%Y%m}.nc"

    if out_file.exists() and not overwrite:
        log.info(f"[ERA5] Đã có: {out_file.name} (bỏ qua, dùng --overwrite để tải lại).")
        return out_file

    # Chuẩn hóa tên biến
    raw_vars = variables or cfg.get("data_sources", {}).get("era5", {}).get(
        "variables", ["tp", "t2m"]
    )
    cds_vars = [ERA5_VAR_MAP.get(v, v) for v in raw_vars]

    years = [str(y) for y in range(start.year, end.year + 1)]
    months = [f"{m:02d}" for m in range(1, 13)]

    log.info(f"[ERA5] Tải {cds_vars} | {start.date()} → {end.date()}")
    log.info(f"[ERA5] bbox = [{lon_min}, {lat_min}, {lon_max}, {lat_max}]")

    # Ưu tiên dùng .cdsapirc trong thư mục dự án (tránh Permission Denied ở thư mục home)
    import os
    _project_rc = Path(__file__).resolve().parents[2] / ".cdsapirc"
    if _project_rc.exists() and "CDSAPI_RC" not in os.environ:
        os.environ["CDSAPI_RC"] = str(_project_rc)
        log.info(f"[ERA5] Dùng credentials: {_project_rc}")

    client = cdsapi.Client()

    # CDS API mới (cdsapi >= 0.7) đôi khi trả về .zip dù request format=netcdf
    # → tải về file tạm, kiểm tra, giải nén nếu cần
    import zipfile, tempfile, shutil

    tmp_path = out_dir / f"_tmp_{out_file.stem}.download"
    client.retrieve(
        "reanalysis-era5-single-levels-monthly-means",
        {
            "product_type": "monthly_averaged_reanalysis",
            "variable": cds_vars,
            "year": years,
            "month": months,
            "time": "00:00",
            "area": [lat_max, lon_min, lat_min, lon_max],  # N, W, S, E
            "format": "netcdf",
        },
        str(tmp_path),
    )

    # Kiểm tra xem có phải file ZIP không
    if zipfile.is_zipfile(tmp_path):
        log.info(f"[ERA5] Phát hiện file ZIP, đang giải nén vào {out_dir} ...")
        with zipfile.ZipFile(tmp_path, 'r') as zf:
            nc_members = [m for m in zf.namelist() if m.endswith('.nc')]
            log.info(f"[ERA5] Các file trong ZIP: {nc_members}")
            zf.extractall(out_dir)
        tmp_path.unlink()  # Xóa file tạm
        log.info(f"[ERA5] Đã giải nén {len(nc_members)} file .nc vào {out_dir}")
        # Trả về None vì không còn 1 file duy nhất nữa
        return out_dir
    else:
        # File hợp lệ → đổi tên về đích
        shutil.move(str(tmp_path), str(out_file))
        log.info(f"[ERA5] Đã lưu: {out_file}")
        return out_file


# ---------------------------------------------------------------------------
# CHIRPS monthly — UCSB
# ---------------------------------------------------------------------------
CHIRPS_MONTHLY_URL = (
    "https://data.chc.ucsb.edu/products/CHIRPS-2.0/"
    "global_monthly/tifs/chirps-v2.0.{year}.{month:02d}.tif.gz"
)
CHIRPS_GLOBAL_NC_URL = (
    "https://data.chc.ucsb.edu/products/CHIRPS-2.0/"
    "global_monthly/netcdf/chirps-v2.0.{year}.monthly.nc"
)


def download_chirps_month_tif(
    year: int, month: int, out_dir: Path, overwrite: bool = False
) -> Optional[Path]:
    """Tải 1 tháng CHIRPS dạng .tif (giải nén từ .gz). Trả về None nếu lỗi."""
    out_dir.mkdir(parents=True, exist_ok=True)
    tif_path = out_dir / f"chirps-v2.0.{year}.{month:02d}.tif"
    if tif_path.exists() and not overwrite:
        log.info(f"[CHIRPS] Đã có: {tif_path.name}")
        return tif_path

    gz_path = out_dir / f"chirps-v2.0.{year}.{month:02d}.tif.gz"
    url = CHIRPS_MONTHLY_URL.format(year=year, month=month)
    try:
        log.info(f"[CHIRPS] Tải {year}-{month:02d}...")
        urllib.request.urlretrieve(url, gz_path)
        with gzip.open(gz_path, "rb") as f_in, open(tif_path, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)
        gz_path.unlink(missing_ok=True)
        return tif_path
    except Exception as e:
        log.warning(f"[CHIRPS] Lỗi {year}-{month:02d}: {e}")
        gz_path.unlink(missing_ok=True)
        return None


def download_chirps_year_nc(
    year: int, out_dir: Path, overwrite: bool = False
) -> Optional[Path]:
    """
    Tải 1 năm CHIRPS dạng NetCDF global (1 file/năm) — NHANH HƠN
    so với tải từng tháng .tif. File ~ 100-200 MB/năm.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"chirps-v2.0.{year}.monthly.nc"
    if out_file.exists() and not overwrite:
        log.info(f"[CHIRPS] Đã có: {out_file.name}")
        return out_file

    url = CHIRPS_GLOBAL_NC_URL.format(year=year)
    try:
        log.info(f"[CHIRPS] Tải năm {year} (NetCDF global)...")
        urllib.request.urlretrieve(url, out_file)
        return out_file
    except Exception as e:
        log.warning(f"[CHIRPS] Lỗi năm {year}: {e}")
        out_file.unlink(missing_ok=True)
        return None


def download_chirps_range(
    cfg: dict, overwrite: bool = False, mode: str = "nc"
) -> list[Path]:
    """
    Tải CHIRPS cho toàn giai đoạn trong config.

    mode='nc'  : tải theo năm, 1 file/năm (khuyến nghị, nhanh)
    mode='tif' : tải theo tháng, 1 file/tháng (chậm, nhiều file)
    """
    start = pd.Timestamp(cfg["time"]["start"])
    end = pd.Timestamp(cfg["time"]["end"])
    out_dir = cfg["paths"]["raw"] / "chirps"

    if mode == "nc":
        years = range(start.year, end.year + 1)
        log.info(f"[CHIRPS] Tải {len(years)} năm (NetCDF) → {out_dir}")
        paths = [p for y in years if (p := download_chirps_year_nc(y, out_dir, overwrite))]
    elif mode == "tif":
        periods = pd.period_range(start, end, freq="M")
        log.info(f"[CHIRPS] Tải {len(periods)} tháng (GeoTIFF) → {out_dir}")
        paths = [
            p
            for pr in periods
            if (p := download_chirps_month_tif(pr.year, pr.month, out_dir, overwrite))
        ]
    else:
        raise ValueError(f"mode không hợp lệ: {mode}")

    log.info(f"[CHIRPS] Hoàn tất: {len(paths)} file.")
    return paths


# ---------------------------------------------------------------------------
# SPI / SPEI có sẵn — ERA5-Drought (Copernicus CDS)
# ---------------------------------------------------------------------------
def download_era5_drought_spi_spei(
    cfg: dict, overwrite: bool = False
) -> Optional[Path]:
    """
    Tải SPI/SPEI toàn cầu đã tính sẵn từ ERA5-Drought (CDS).
    Rất tiện khi không muốn tự tính; dữ liệu có sẵn từ 1940.

    Dataset: 'derived-drought-historical-monthly'
    Tham khảo: https://cds.climate.copernicus.eu/
    """
    try:
        import cdsapi
    except ImportError:
        raise ImportError("Chưa cài cdsapi. Chạy `pip install cdsapi`.")

    region = cfg["region"]
    lon_min, lat_min, lon_max, lat_max = region["bbox"]
    start = pd.Timestamp(cfg["time"]["start"])
    end = pd.Timestamp(cfg["time"]["end"])

    out_dir = cfg["paths"]["raw"] / "spi_spei_ready"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = (
        out_dir
        / f"era5_drought_{region['name']}_{start:%Y%m}_{end:%Y%m}.nc"
    )
    if out_file.exists() and not overwrite:
        log.info(f"[ERA5-Drought] Đã có: {out_file.name}")
        return out_file

    # Lấy danh sách scale + index từ config
    index_cfg = cfg.get("index", {})
    scales = index_cfg.get("scales", [1, 3, 6, 12])
    index_type = index_cfg.get("type", "SPEI").lower()  # 'spi' | 'spei'

    # Dataset ERA5-Drought dùng tên biến dạng 'spi_1', 'spei_3', ...
    variables = [f"{index_type}_{s}" for s in scales]

    years = [str(y) for y in range(start.year, end.year + 1)]
    months = [f"{m:02d}" for m in range(1, 13)]

    log.info(f"[ERA5-Drought] Tải {variables} | {start.date()} → {end.date()}")

    # Ưu tiên dùng .cdsapirc trong thư mục dự án (tránh Permission Denied ở thư mục home)
    import os
    _project_rc = Path(__file__).resolve().parents[2] / ".cdsapirc"
    if _project_rc.exists() and "CDSAPI_RC" not in os.environ:
        os.environ["CDSAPI_RC"] = str(_project_rc)
        log.info(f"[ERA5] Dùng credentials: {_project_rc}")

    client = cdsapi.Client()
    client.retrieve(
        "derived-drought-historical-monthly",
        {
            "variable": variables,
            "year": years,
            "month": months,
            "area": [lat_max, lon_min, lat_min, lon_max],
            "format": "netcdf",
        },
        str(out_file),
    )
    log.info(f"[ERA5-Drought] Đã lưu: {out_file}")
    return out_file


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Tải dữ liệu thô cho drought-spi-spei-vn"
    )
    p.add_argument(
        "source",
        choices=["era5", "chirps", "spi-spei", "all"],
        help="Nguồn dữ liệu cần tải",
    )
    p.add_argument("--config", default="config/config.yml")
    p.add_argument("--overwrite", action="store_true", help="Tải lại file đã có")
    p.add_argument(
        "--chirps-mode",
        choices=["nc", "tif"],
        default="nc",
        help="'nc': 1 file/năm (nhanh) | 'tif': 1 file/tháng (chậm)",
    )
    return p


def main(argv: Optional[list[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    cfg = load_config(args.config)

    if args.source in ("era5", "all"):
        download_era5_monthly(cfg, overwrite=args.overwrite)

    if args.source in ("chirps", "all"):
        download_chirps_range(
            cfg, overwrite=args.overwrite, mode=args.chirps_mode
        )

    if args.source in ("spi-spei", "all"):
        download_era5_drought_spi_spei(cfg, overwrite=args.overwrite)

    log.info("Hoàn tất tải dữ liệu.")
    return 0


if __name__ == "__main__":
    sys.exit(main())