"""
src/utils/auto_update_config.py
===============================
Tự động cập nhật ngày kết thúc (time.end) trong file config.yaml
thành ngày cuối cùng của tháng trước (để đảm bảo dữ liệu ERA5 đã có sẵn).
"""
import logging
from datetime import datetime, timedelta
import yaml
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
log = logging.getLogger(__name__)

def update_config_to_latest_date(config_path="config/config.yml"):
    # 1. Tính toán ngày cuối cùng của tháng trước
    today = datetime.today()
    first_day_this_month = today.replace(day=1)
    last_day_prev_month = first_day_this_month - timedelta(days=1)
    latest_date_str = last_day_prev_month.strftime("%Y-%m-%d")

    # 2. Đọc file config hiện tại
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    old_end = cfg['time']['end']
    
    # 3. Cập nhật ngày mới
    cfg['time']['end'] = latest_date_str

    # 4. Ghi lại vào file
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.dump(cfg, f, sort_keys=False, allow_unicode=True, default_flow_style=False)

    log.info(f"Đã tự động cập nhật config: time.end từ '{old_end}' thành '{latest_date_str}'")
    return latest_date_str

if __name__ == "__main__":
    update_config_to_latest_date()