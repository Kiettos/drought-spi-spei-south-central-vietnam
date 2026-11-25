"""
src/models/evaluate.py
======================
Đánh giá mô hình: Vẽ Feature Importance và Bản đồ dự đoán hạn hán.
"""
import logging
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns

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

def evaluate_model(cfg):
    data_dir = cfg['paths']['processed']
    model_dir = cfg['paths']['models']
    fig_dir = cfg['paths']['outputs'] / "figures"
    
    # 1. Load dữ liệu và mô hình
    log.info("Đang tải mô hình và dữ liệu Test...")
    test_df = pd.read_csv(data_dir / "test_dataset.csv")
    features = pd.read_csv(model_dir / "features_list.csv", header=None)[0].tolist()
    
    model = xgb.XGBClassifier()
    model.load_model(model_dir / "xgboost_drought_model.json")
    
    # ==========================================
    # BIỂU ĐỒ 1: FEATURE IMPORTANCE
    # ==========================================
    log.info("Đang vẽ biểu đồ Feature Importance...")
    plt.figure(figsize=(10, 6))
    # Lấy 15 features quan trọng nhất
    importances = model.feature_importances_
    indices = np.argsort(importances)[-15:]
    
    plt.title('Top 15 yếu tố ảnh hưởng đến dự đoán hạn hán', fontsize=14)
    plt.barh(range(len(indices)), importances[indices], color='teal', align='center')
    plt.yticks(range(len(indices)), [features[i] for i in indices])
    plt.xlabel('Mức độ quan trọng (Feature Importance)')
    plt.tight_layout()
    plt.savefig(fig_dir / "feature_importance.png", dpi=300)
    plt.close()
    
    # ==========================================
    # BIỂU ĐỒ 2: BẢN ĐỒ DỰ ĐOÁN (Chọn 1 tháng hạn nặng để vẽ)
    # ==========================================
    log.info("Đang vẽ bản đồ dự đoán...")
    # Dự đoán xác suất hạn hán cho toàn bộ tập Test
    test_df['pred_proba'] = model.predict_proba(test_df[features])[:, 1]
    
    # Tìm một tháng bất kỳ trong tập Test (Ví dụ: Đầu năm 2020)
    # Chuyển cột time về dạng datetime để lọc
    test_df['time'] = pd.to_datetime(test_df['time'])
    sample_time = test_df[test_df['time'] == '2020-04-01'] # Đổi tháng này tùy ý
    
    if len(sample_time) > 0:
        # Pivot bảng dữ liệu ra dạng lưới 2D (Vĩ độ x Kinh độ)
        grid_pred = sample_time.pivot(index='lat', columns='lon', values='pred_proba')
        grid_actual = sample_time.pivot(index='lat', columns='lon', values='target') # SPEI_3 thực tế
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Vẽ Xác suất AI dự đoán
        sns.heatmap(grid_pred, ax=axes[0], cmap='YlOrRd', cbar_kws={'label': 'Xác suất hạn hán'})
        axes[0].set_title('AI dự đoán (Xác suất hạn)')
        axes[0].invert_yaxis()
        
        # Vẽ Thực tế (SPEI_3)
        sns.heatmap(grid_actual, ax=axes[1], cmap='RdYlBu', center=0, cbar_kws={'label': 'Chỉ số SPEI-3'})
        axes[1].set_title('Thực tế (SPEI-3)')
        axes[1].invert_yaxis()
        
        plt.suptitle(f'So sánh dự đoán và thực tế - Tháng {sample_time["time"].iloc[0].strftime("%m/%Y")}', fontsize=16)
        plt.tight_layout()
        plt.savefig(fig_dir / "spatial_map_prediction.png", dpi=300)
        plt.close()
        log.info(f"Đã lưu bản đồ tại: {fig_dir}")
    else:
        log.warning("Không tìm thấy dữ liệu cho tháng 2020-04-01 để vẽ bản đồ.")

if __name__ == "__main__":
    cfg = load_config()
    evaluate_model(cfg)
    log.info("Hoàn tất đánh giá và trực quan hóa.")