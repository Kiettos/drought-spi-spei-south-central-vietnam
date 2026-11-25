"""
src/utils/log_setup.py
======================
Cấu hình logging dùng chung cho mọi bước pipeline.

Vì sao cần: mặc định các script chỉ log ra console bằng logging.basicConfig,
nên không có file log nào được giữ lại. Ngoài ra khi redirect stdout ra file,
Windows dùng codec cp1252 và các dòng tiếng Việt ("Bình thường", "Hạn hán")
làm chương trình chết với UnicodeEncodeError.

Hàm setup_logging() giải quyết cả hai:
    1. Ép stdout/stderr sang UTF-8.
    2. Ghi log ra cả console và logs/<name>_<timestamp>.log (UTF-8).
"""

import logging
import sys
from datetime import datetime
from pathlib import Path

LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def _force_utf8_console():
    """Ép stdout/stderr sang UTF-8 để không chết khi redirect ra file."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass  # stream không hỗ trợ (vd: bị thay bằng object khác)


def setup_logging(name, log_dir="logs", level=logging.INFO):
    """
    Cấu hình root logger ghi ra console + file.

    name    : tên dùng để đặt file log, vd "train", "evaluate", "pipeline"
    log_dir : thư mục chứa log (mặc định logs/)
    Trả về  : Path của file log vừa tạo.
    """
    _force_utf8_console()

    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"{name}_{datetime.now():%Y%m%d_%H%M%S}.log"

    formatter = logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT)

    root = logging.getLogger()
    root.setLevel(level)
    for handler in root.handlers[:]:
        root.removeHandler(handler)

    console = logging.StreamHandler(stream=sys.stdout)
    console.setFormatter(formatter)
    root.addHandler(console)

    file_handler = logging.FileHandler(log_file, mode="w", encoding="utf-8")
    file_handler.setFormatter(formatter)
    root.addHandler(file_handler)

    logging.getLogger(__name__).info(f"Ghi log vào file: {log_file}")
    return log_file
