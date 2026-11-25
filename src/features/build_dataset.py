"""
src/features/build_dataset.py
=============================
Chuyển dữ liệu từ dạng lưới 3D (NetCDF) sang dạng bảng 2D (Tabular).
Tạo các đặc trưng (features) từ quá khứ và nhãn (target) cho tương lai.
Lưu thành file CSV để đưa vào mô hình Machine Learning.
"""
import logging
import sys
from pathlib import Path
import pandas as pd
import xarray as xr

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

def build_dataset(cfg):
    region_name = cfg['region']['name']
    era5_file = cfg['paths']['interim'] / f"era5_processed_{region_name}.nc"
    indices_file = cfg['paths']['processed'] / f"indices_{region_name}.nc"
    
    if not era5_file.exists() or not indices_file.exists():
        log.error("Thiếu file dữ liệu. Hãy chạy preprocess và spi_spei trước.")
        sys.exit(1)

    log.info("Đang gộp dữ liệu thời tiết và chỉ số hạn hán...")
    ds_era5 = xr.open_dataset(era5_file)
    ds_indices = xr.open_dataset(indices_file)
    
    # Gộp 2 dataset theo thời gian, vĩ độ, kinh độ
    ds = xr.merge([ds_era5, ds_indices])
    
    log.info("Chuyển đổi sang dạng bảng (DataFrame)... (Bước này tốn vài giây)")
    df = ds.to_dataframe().reset_index()
    
    # Sắp xếp theo tọa độ và thời gian để tạo lag chính xác
    df = df.sort_values(by=['lat', 'lon', 'time']).reset_index(drop=True)
    
    log.info("Tạo đặc trưng (Lag features) và Nhãn (Target)...")
    
    # Chọn chỉ số hạn làm mục tiêu dự đoán (Ví dụ: SPEI_3)
    target_index = 'SPEI_3'
    lead_time = 1 # Dự đoán trước 1 tháng
    
    # Hàm tạo lag cho từng điểm lưới (từng pixel)
    def create_features(group):
        # 1. Tạo Target: Dịch chuyển thời gian lên `lead_time` tháng (Tương lai)
        group['target'] = group[target_index].shift(-lead_time)
        
        # 2. Tạo nhãn phân loại (1: Hạn, 0: Bình thường/Không hạn). Hạn khi SPEI <= -1.0
        group['is_drought'] = (group['target'] <= -1.0).astype(int)
        
        # 3. Tạo Lag features: Lấy dữ liệu mưa, nhiệt độ, chỉ số của các tháng trước
        for lag in [1, 2, 3]:
            group[f'tp_lag{lag}'] = group['tp'].shift(lag)
            group[f't2m_lag{lag}'] = group['t2m'].shift(lag)
            group[f'spei3_lag{lag}'] = group[target_index].shift(lag)
            
        return group

    # Áp dụng hàm cho từng điểm lưới (nhóm theo lat, lon)
    df = df.groupby(['lat', 'lon'], group_keys=False).apply(create_features)
    
    # Xóa các dòng có chứa NaN (do bị trễ ở đầu chuỗi hoặc cuối chuỗi)
    df = df.dropna()
    
    log.info(f"Kích thước dữ liệu sau khi tạo features: {df.shape}")
    
    # 4. Chia tập Train/Val/Test theo thời gian (KHÔNG chia ngẫu nhiên)
    # 1984-2013 (30 năm): Train
    # 2014-2018 (5 năm): Validation
    # 2019-2023 (5 năm): Test
    train_df = df[df['time'].dt.year <= 2013]
    val_df = df[(df['time'].dt.year >= 2014) & (df['time'].dt.year <= 2018)]
    test_df = df[df['time'].dt.year >= 2019]
    
    log.info(f"Train: {len(train_df)} mẫu | Val: {len(val_df)} mẫu | Test: {len(test_df)} mẫu")
    
    # Lưu ra file CSV
    out_dir = cfg['paths']['processed']
    train_df.to_csv(out_dir / "train_dataset.csv", index=False)
    val_df.to_csv(out_dir / "val_dataset.csv", index=False)
    test_df.to_csv(out_dir / "test_dataset.csv", index=False)
    
    log.info(f"Đã lưu 3 file CSV tại: {out_dir}")

if __name__ == "__main__":
    cfg = load_config()
    build_dataset(cfg)
    log.info("Hoàn tất tạo dataset cho Machine Learning.")