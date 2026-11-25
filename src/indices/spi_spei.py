"""
src/indices/spi_spei.py
=======================
Tính toán chỉ số hạn hán SPI và SPEI từ dữ liệu ERA5 đã tiền xử lý.
"""
import logging
import sys
from pathlib import Path
from climate_indices import indices, pet_thornthwaite
from climate_indices.compute import Periodicity  # <-- Thêm dòng này

import numpy as np
import pandas as pd
import xarray as xr
from climate_indices import indices, pet_thornthwaite

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
log = logging.getLogger(__name__)

def load_config(path="config/config.yml"):
    import yaml
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    for k, v in cfg.get("paths", {}).items():
        cfg["paths"][k] = Path(v)
        cfg["paths"][k].mkdir(parents=True, exist_ok=True)
    return cfg

def compute_indices(cfg):
    region_name = cfg['region']['name']
    in_file = cfg['paths']['interim'] / f"era5_processed_{region_name}.nc"
    
    if not in_file.exists():
        log.error(f"Không tìm thấy file: {in_file}")
        sys.exit(1)

    log.info(f"Đọc dữ liệu từ: {in_file.name}")
    ds = xr.open_dataset(in_file)
    
    # Giai đoạn làm chuẩn (thường lấy 30 năm đầu tiên)
    start_year = int(pd.to_datetime(ds.time[0].values).year)
    cal_start = start_year
    cal_end = start_year + 29
    log.info(f"Giai đoạn làm chuẩn (calibration): {cal_start} - {cal_end}")

    scales = cfg.get('index', {}).get('scales', [1, 3, 6, 12])
    out_ds = xr.Dataset()
    
    lat_mean = float(ds['lat'].mean().values)

    # Hàm tính PET cho từng điểm lưới
    def calc_pet_1d(t2m_series):
        if np.all(np.isnan(t2m_series)):
            return np.full_like(t2m_series, np.nan)
        # Tính PET bằng Thornthwaite (mm/tháng)
        return pet_thornthwaite(
            t2m_series,
            latitude=lat_mean,
            data_start_year=cal_start
        )

    log.info("Đang tính PET (Bốc hơi tiềm năng)...")
    pet_da = xr.apply_ufunc(
        calc_pet_1d, ds['t2m'],
        input_core_dims=[["time"]],
        output_core_dims=[["time"]],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[float]
    )

    # Hàm tính SPI cho từng điểm lưới
    def calc_spi_1d(precip_series, scale):
        if np.all(np.isnan(precip_series)):
            return np.full_like(precip_series, np.nan)
        return indices.spi(
            values=precip_series,
            scale=scale,
            distribution=indices.Distribution.gamma,
            data_start_year=cal_start,
            calibration_year_initial=cal_start,
            calibration_year_final=cal_end,
            periodicity=Periodicity.monthly  # <-- Sửa ở đây
    )

    # Hàm tính SPEI cho từng điểm lưới
    def calc_spei_1d(precip_series, pet_series, scale):
        if np.all(np.isnan(precip_series)) or np.all(np.isnan(pet_series)):
            return np.full_like(precip_series, np.nan)
        return indices.spei(
            precips_mm=precip_series,
            pet_mm=pet_series,
            scale=scale,
            distribution=indices.Distribution.loglogistic,
            data_start_year=cal_start,
            calibration_year_initial=cal_start,
            calibration_year_final=cal_end,
            periodicity=Periodicity.monthly  # <-- Sửa ở đây
    )

    for scale in scales:
        # 1. Tính SPI
        log.info(f"  -> Đang tính SPI-{scale} ...")
        spi_da = xr.apply_ufunc(
            calc_spi_1d, ds['tp'],
            kwargs={'scale': scale},
            input_core_dims=[["time"]], output_core_dims=[["time"]],
            vectorize=True, dask="parallelized", output_dtypes=[float]
        )
        spi_da.name = f"SPI_{scale}"
        out_ds[spi_da.name] = spi_da

        # 2. Tính SPEI
        log.info(f"  -> Đang tính SPEI-{scale} ...")
        spei_da = xr.apply_ufunc(
            calc_spei_1d, ds['tp'], pet_da,
            kwargs={'scale': scale},
            input_core_dims=[["time"], ["time"]], output_core_dims=[["time"]],
            vectorize=True, dask="parallelized", output_dtypes=[float]
        )
        spei_da.name = f"SPEI_{scale}"
        out_ds[spei_da.name] = spei_da

    # Lưu file
    out_file = cfg['paths']['processed'] / f"indices_{region_name}.nc"
    out_ds.to_netcdf(out_file)
    log.info(f"Đã lưu kết quả SPI/SPEI tại: {out_file}")

if __name__ == "__main__":
    cfg = load_config()
    compute_indices(cfg)
    log.info("Hoàn tất tính toán chỉ số hạn hán.")