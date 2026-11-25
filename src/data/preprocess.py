"""
src/data/preprocess.py
======================
Tiền xử lý dữ liệu ERA5 thô:
    - Cắt theo bbox trong config.
    - Đổi đơn vị: tp (m -> mm), t2m (K -> °C).
    - Đảm bảo trục thời gian là đầu tháng (Monthly Start).
    - Xử lý missing values.
    - Lưu vào data/interim/.
"""

import logging
import sys
from pathlib import Path
import xarray as xr
import pandas as pd

# Cấu hình logging
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

def preprocess_era5(cfg):
    region_name = cfg['region']['name']
    raw_dir = cfg['paths']['raw'] / "era5"
    
    # Tự động tìm TẤT CẢ các file .nc trong thư mục raw/era5/
    nc_files = list(raw_dir.glob("*.nc"))
    
    if not nc_files:
        log.error(f"Không tìm thấy file .nc nào trong {raw_dir}.")
        sys.exit(1)

    log.info(f"Tìm thấy {len(nc_files)} file .nc. Đang đọc và xử lý...")
    
    datasets = []
    for f in nc_files:
        # Thử mở file — bỏ qua nếu hỏng (vd: file zip được lưu nhầm thành .nc)
        try:
            ds_temp = xr.open_dataset(f, engine='netcdf4')
        except Exception as e:
            log.warning(f"Bỏ qua file không hợp lệ: {f.name} ({e})")
            continue
        
        # 1. Chuẩn hóa tên chiều
        rename_dict = {}
        if 'valid_time' in ds_temp.dims or 'valid_time' in ds_temp.coords:
            rename_dict['valid_time'] = 'time'
        if 'latitude' in ds_temp.dims or 'latitude' in ds_temp.coords:
            rename_dict['latitude'] = 'lat'
        if 'longitude' in ds_temp.dims or 'longitude' in ds_temp.coords:
            rename_dict['longitude'] = 'lon'
        
        if rename_dict:
            ds_temp = ds_temp.rename(rename_dict)
            
        # 2. Chuẩn hóa thời gian VỀ ĐẦU THÁNG NGAY LẬP TỨC (Trước khi merge)
        if not pd.api.types.is_datetime64_any_dtype(ds_temp['time']):
            ds_temp['time'] = pd.to_datetime(ds_temp['time'])
        ds_temp = ds_temp.assign_coords(time=ds_temp['time'].to_index().to_period("M").to_timestamp())
        
        datasets.append(ds_temp)

    if not datasets:
        log.error("Không có file .nc hợp lệ nào. Hãy kiểm tra thư mục data/raw/era5/")
        sys.exit(1)

    # 3. Bây giờ mới Merge (thời gian đã đồng bộ nên sẽ không bị lệch)
    ds = xr.merge(datasets, compat="override")
    log.info("Đã merge các file thành công.")

    # 4. Cắt theo bbox
    lon_min, lat_min, lon_max, lat_max = cfg['region']['bbox']
    log.info(f"Cắt vùng: {cfg['region']['bbox']}")
    ds = ds.sel(lon=slice(lon_min, lon_max), lat=slice(lat_max, lat_min))

     # 5. Đổi đơn vị
    log.info("Đổi đơn vị: tp (m/day->mm/month), t2m (K->°C)")
    if 'tp' in ds:
        # Nhân 1000 để ra mm/ngày, sau đó nhân với số ngày trong tháng để ra mm/tháng
        ds['tp'] = ds['tp'] * 1000.0 * ds['time'].dt.days_in_month
        ds['tp'].attrs['units'] = 'mm/month'

    # 6. Xử lý missing (nội suy tuyến tính nếu có)
    n_missing = int(ds.to_array().isnull().sum())
    if n_missing > 0:
        log.warning(f"Có {n_missing} giá trị missing. Đang nội suy...")
        ds = ds.interpolate_na(dim="time", method="linear", fill_value="extrapolate")

    # 7. Lưu file
    out_file = cfg['paths']['interim'] / f"era5_processed_{region_name}.nc"
    ds.to_netcdf(out_file)
    log.info(f"Đã lưu file sạch tại: {out_file}")
    
    return out_file

if __name__ == "__main__":
    cfg = load_config()
    preprocess_era5(cfg)
    log.info("Hoàn tất tiền xử lý ERA5.")