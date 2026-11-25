"""
run_pipeline.py
===============
Tự động chạy toàn bộ Pipeline:
1. Tự động cập nhật Config (ngày mới nhất)
2. Tải dữ liệu mới (download)
3. Tiền xử lý (preprocess)
4. Tính chỉ số SPI/SPEI
5. Tạo lại Dataset cho ML
6. Train lại mô hình
7. Đánh giá mô hình
"""
import subprocess
import sys
import time
import logging
from src.utils.auto_update_config import update_config_to_latest_date

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
log = logging.getLogger(__name__)

def run_step(step_name: str, command: list):
    log.info(f"========== BẮT ĐẦU: {step_name} ==========")
    start_time = time.time()
    
    # Chạy lệnh python -m <module>
    result = subprocess.run(command, text=True)
    
    if result.returncode != 0:
        log.error(f"LỖI tại bước: {step_name}. Dừng toàn bộ pipeline.")
        sys.exit(1)
        
    elapsed = time.time() - start_time
    log.info(f"========== HOÀN THÀNH: {step_name} (Mất {elapsed:.1f} giây) ==========\n")

if __name__ == "__main__":
    log.info("🚀 BẮT ĐẦU CHẠY TỰ ĐỘNG TOÀN BỘ PIPELINE 🚀")
    
    # 1. Tự động cập nhật Config
    log.info("Bước 0: Cập nhật thời gian mới nhất vào config...")
    update_config_to_latest_date("config/config.yml")
    
    # Danh sách các bước cần chạy
    # Danh sách các bước cần chạy (Dùng sys.executable để lấy đúng Python của .venv)
    steps = [
        ("Tải dữ liệu ERA5", [sys.executable, "-m", "src.data.download", "era5"]),
        ("Tiền xử lý dữ liệu", [sys.executable, "-m", "src.data.preprocess"]),
        ("Tính toán SPI/SPEI", [sys.executable, "-m", "src.indices.spi_spei"]),
        ("Xây dựng Dataset cho ML", [sys.executable, "-m", "src.features.build_dataset"]),
        ("Huấn luyện mô hình AI", [sys.executable, "-m", "src.modelS.train"]),
        ("Đánh giá và xuất biểu đồ", [sys.executable, "-m", "src.modelS.evaluate"])
    ]
    
    # Chạy từng bước
    for step_name, command in steps:
        run_step(step_name, command)
        
    log.info("🎉 TẤT CẢ CÁC BƯỚC ĐÃ CHẠY THÀNH CÔNG! BẠN CÓ THỂ MỞ APP ĐỂ XEM KẾT QUẢ.")