"""
src/models/train.py
===================
Huấn luyện mô hình dự đoán hạn hán (XGBoost).
Sử dụng dữ liệu đã tạo từ bước build_dataset.
Đánh giá trên tập Test và lưu mô hình.
"""
import logging
import sys
from pathlib import Path
import joblib
import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

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

def train_model(cfg):
    data_dir = cfg['paths']['processed']
    
    log.info("Đang đọc dữ liệu Train/Val/Test...")
    try:
        train_df = pd.read_csv(data_dir / "train_dataset.csv")
        val_df = pd.read_csv(data_dir / "val_dataset.csv")
        test_df = pd.read_csv(data_dir / "test_dataset.csv")
    except FileNotFoundError:
        log.error("Không tìm thấy file CSV. Hãy chạy build_dataset.py trước.")
        sys.exit(1)

    # Chọn các cột làm đặc trưng (Features)
    # Bỏ các cột không dùng để dự đoán: thời gian, tọa độ, nhãn mục tiêu
    features = [c for c in train_df.columns if c not in ['time', 'lat', 'lon', 'target', 'is_drought']]
    
    X_train, y_train = train_df[features], train_df['is_drought']
    X_val, y_val = val_df[features], val_df['is_drought']
    X_test, y_test = test_df[features], test_df['is_drought']

    log.info(f"Số lượng đặc trưng (Features) sử dụng: {len(features)}")
    log.info(f"Tỷ lệ hạn hán trong tập Train: {y_train.mean()*100:.2f}%")

    # Khởi tạo mô hình XGBoost
    log.info("Đang huấn luyện mô hình XGBoost...")
    # Dùng scale_pos_weight để xử lý mất cân bằng dữ liệu (số ngày không hạn nhiều hơn ngày hạn)
    scale_pos_weight = (len(y_train) - y_train.sum()) / y_train.sum()
    
    model = XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.1,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        n_jobs=-1,
        eval_metric='logloss'
    )

    # Train với tập Train, đánh giá chéo trên tập Validation
    model.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_val, y_val)],
        verbose=False
    )

    # --- ĐÁNH GIÁ MÔ HÌNH ---
    log.info("Đang đánh giá mô hình trên tập Test (Dữ liệu chưa từng thấy)...")
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    log.info("\n--- KẾT QUẢ ĐÁNH GIÁ ---")
    log.info(f"ROC-AUC Score: {roc_auc_score(y_test, y_pred_proba):.4f}")
    
    log.info("\nBáo cáo phân loại (Classification Report):")
    report = classification_report(y_test, y_pred, target_names=["Bình thường (0)", "Hạn hán (1)"])
    print(report)

    log.info("Ma trận nhầm lẫn (Confusion Matrix):")
    print(confusion_matrix(y_test, y_pred))

    # Lưu mô hình
    model_dir = cfg['paths']['models']
    model_path = model_dir / "xgboost_drought_model.json"
    model.save_model(model_path)
    log.info(f"Đã lưu mô hình tại: {model_path}")

    # Lưu danh sách features để dùng cho bước dự đoán sau này
    pd.Series(features).to_csv(model_dir / "features_list.csv", index=False, header=False)

if __name__ == "__main__":
    cfg = load_config()
    train_model(cfg)
    log.info("Hoàn tất huấn luyện.")