"""
src/data/preprocess.py
======================
Tiền xử lý dữ liệu ERA5 thô:
    - Polygon clipping: cắt theo ranh giới các tỉnh (bỏ biển + nước láng giềng).
    - Đổi đơn vị: tp (m -> mm/tháng), t2m (K -> °C).
    - Đảm bảo trục thời gian là đầu tháng (Monthly Start).
    - Xử lý missing values (chỉ nội suy theo thời gian, không lấp qua biển).
    - Lưu vào data/interim/.

Cách dùng:
    python -m src.data.preprocess
    python -m src.data.preprocess --region nam_trung_bo
"""

import argparse
import logging
import sys
import unicodedata
from pathlib import Path

import pandas as pd
import xarray as xr

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


# ---------------------------------------------------------------------------
# Đối chiếu tên tỉnh
# ---------------------------------------------------------------------------
def _norm_name(s):
    """
    Chuẩn hoá tên tỉnh để so khớp.

    GADM 4.1 lưu tên có dấu và BỎ KHOẢNG TRẮNG ("ĐàNẵng", "QuảngNam"),
    trong khi người dùng thường viết "Da Nang" / "Quang Nam".
    Hàm này đưa cả hai về cùng một dạng: 'danang', 'quangnam'.
    """
    s = str(s).replace("Đ", "D").replace("đ", "d")   # NFKD không tách được Đ
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    return s.replace(" ", "").replace("-", "").replace(".", "").replace("_", "").lower()


def _find_map_file(cfg):
    """Tìm file bản đồ: ưu tiên config, sau đó vài vị trí quen thuộc."""
    declared = cfg["region"].get("map_file")
    if declared:
        p = Path(declared)
        return p if p.exists() else None

    candidates = [
        Path("config/gadm41_VNM_1.json"),
        Path("data/vietnam_map.json"),
        Path("data/gadm41_VNM_1.json"),
    ]
    candidates += sorted(Path("config").glob("*VNM*.json"))
    candidates += sorted(Path("data").glob("*VNM*.json"))
    for p in candidates:
        if p.exists():
            return p
    return None


# ---------------------------------------------------------------------------
# Polygon clipping
# ---------------------------------------------------------------------------
def clip_to_region(ds, cfg):
    """
    Cắt dataset theo ranh giới các tỉnh trong cfg['region']['provinces'].

    Trả về (ds_đã_cắt, mask). mask là None nếu không cấu hình provinces.

    Khác với bản cũ: mọi lỗi cấu hình (thiếu file, thiếu tỉnh, mask rỗng)
    đều dừng chương trình thay vì âm thầm quay lại cắt bbox — vì quay lại
    bbox nghĩa là bản đồ vẫn còn biển và nước láng giềng.
    """
    region = cfg["region"]
    provinces = region.get("provinces") or []

    if not provinces:
        log.warning("region.provinces trống -> bỏ qua polygon clipping, cắt theo bbox.")
        lon_min, lat_min, lon_max, lat_max = region["bbox"]
        return ds.sel(lon=slice(lon_min, lon_max), lat=slice(lat_max, lat_min)), None

    try:
        import geopandas as gpd
        import regionmask
    except ImportError as e:
        log.error(f"Thiếu thư viện địa không gian ({e}). Chạy: pip install geopandas regionmask")
        sys.exit(1)

    map_path = _find_map_file(cfg)
    if map_path is None:
        log.error(f"Không tìm thấy file bản đồ (region.map_file={region.get('map_file')!r}).")
        log.error("Tải GADM Việt Nam level 1 (GeoJSON) và đặt region.map_file trong config/config.yml")
        sys.exit(1)

    log.info(f"Đọc bản đồ: {map_path}")
    vn_map = gpd.read_file(map_path)

    # 1. CRS phải là WGS84, nếu không mask sẽ lệch
    if vn_map.crs is None:
        log.error("Bản đồ không có CRS. Không thể xác định hệ toạ độ.")
        sys.exit(1)
    if vn_map.crs.to_epsg() != 4326:
        log.warning(f"Bản đồ CRS={vn_map.crs} -> chuyển về EPSG:4326")
        vn_map = vn_map.to_crs(4326)

    # 2. Tên cột chứa tên tỉnh (GADM 4.1 = NAME_1, các bản khác có thể khác)
    name_col = next(
        (c for c in ("NAME_1", "NAME", "ten_tinh", "Name", "name", "province") if c in vn_map.columns),
        None,
    )
    if name_col is None:
        log.error(f"Không thấy cột tên tỉnh. Các cột có: {list(vn_map.columns)}")
        sys.exit(1)
    log.info(f"Dùng cột '{name_col}' để lọc tỉnh.")

    # 3. Lọc tỉnh — so khớp sau khi chuẩn hoá dấu và khoảng trắng
    vn_map = vn_map.assign(_key=vn_map[name_col].map(_norm_name))
    wanted = {_norm_name(p): p for p in provinces}
    sub = vn_map[vn_map["_key"].isin(wanted)].reset_index(drop=True)

    found = set(sub["_key"])
    missing = [orig for key, orig in wanted.items() if key not in found]
    if missing:
        log.error(f"Không khớp được {len(missing)}/{len(provinces)} tỉnh: {missing}")
        log.error(f"Tên tỉnh có trong bản đồ: {sorted(vn_map[name_col].astype(str))}")
        sys.exit(1)

    log.info(f"Khớp {len(sub)}/{len(provinces)} tỉnh: {sub[name_col].tolist()}")

    # 4. Cắt thô theo bbox lấy từ chính polygon (không dùng bbox trong config,
    #    vì bbox đó thường rộng hơn nhiều và kéo theo ô biển)
    lon_min, lat_min, lon_max, lat_max = sub.total_bounds
    pad = 0.01
    ds = ds.sel(
        lon=slice(lon_min - pad, lon_max + pad),
        lat=slice(lat_max + pad, lat_min - pad),
    )

    # 5. Mask theo ranh giới hành chính
    mask = regionmask.mask_geopandas(sub, ds["lon"], ds["lat"])

    n_box = ds.sizes["lat"] * ds.sizes["lon"]
    n_keep = int(mask.notnull().sum())
    if n_keep == 0:
        log.error("Mask rỗng: không có ô lưới nào nằm trong ranh giới tỉnh.")
        log.error("Kiểm tra lại toạ độ dataset và danh sách tỉnh.")
        sys.exit(1)

    log.info(
        f"Polygon clipping: {n_box} ô (bbox) -> {n_keep} ô (trong ranh giới tỉnh), "
        f"loại {n_box - n_keep} ô biển / ngoài vùng"
    )
    return ds.where(mask.notnull()), mask


# ---------------------------------------------------------------------------
# Pipeline chính
# ---------------------------------------------------------------------------
def preprocess_era5(cfg):
    region_name = cfg["region"]["name"]
    raw_dir = cfg["paths"]["raw"] / "era5"

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
            ds_temp = xr.open_dataset(f, engine="netcdf4")
        except Exception as e:
            log.warning(f"Bỏ qua file không hợp lệ: {f.name} ({e})")
            continue

        # 1. Chuẩn hóa tên chiều
        rename_dict = {}
        if "valid_time" in ds_temp.dims or "valid_time" in ds_temp.coords:
            rename_dict["valid_time"] = "time"
        if "latitude" in ds_temp.dims or "latitude" in ds_temp.coords:
            rename_dict["latitude"] = "lat"
        if "longitude" in ds_temp.dims or "longitude" in ds_temp.coords:
            rename_dict["longitude"] = "lon"

        if rename_dict:
            ds_temp = ds_temp.rename(rename_dict)

        # 2. Chuẩn hóa thời gian VỀ ĐẦU THÁNG NGAY LẬP TỨC (Trước khi merge)
        if not pd.api.types.is_datetime64_any_dtype(ds_temp["time"]):
            ds_temp["time"] = pd.to_datetime(ds_temp["time"])
        ds_temp = ds_temp.assign_coords(time=ds_temp["time"].to_index().to_period("M").to_timestamp())

        datasets.append(ds_temp)

    if not datasets:
        log.error("Không có file .nc hợp lệ nào. Hãy kiểm tra thư mục data/raw/era5/")
        sys.exit(1)

    # 3. Bây giờ mới Merge (thời gian đã đồng bộ nên sẽ không bị lệch)
    ds = xr.merge(datasets, compat="override")
    log.info("Đã merge các file thành công.")

    # 4. POLYGON CLIPPING — cắt theo ranh giới tỉnh
    ds, mask = clip_to_region(ds, cfg)

    # 5. Đổi đơn vị
    log.info("Đổi đơn vị: tp (m -> mm/tháng), t2m (K -> °C)")
    if "tp" in ds:
        # ERA5 tp là m/ngày -> mm/ngày -> mm/tháng
        ds["tp"] = ds["tp"] * 1000.0 * ds["time"].dt.days_in_month
        ds["tp"].attrs["units"] = "mm/month"

    if "t2m" in ds:
        # QUAN TRỌNG: chưa trừ 273.15 thì t2m vẫn là Kelvin (~300 K).
        # pet_thornthwaite() cần °C, nếu không PET sẽ sai hoàn toàn.
        ds["t2m"] = ds["t2m"] - 273.15
        ds["t2m"].attrs["units"] = "degC"

    # 6. Lưu mask lại để các bước sau biết ô nào hợp lệ
    if mask is not None:
        ds["region"] = mask.astype("float32")
        ds["region"].attrs["long_name"] = "region index (NaN = ngoài ranh giới)"

    # 7. Xử lý missing — CHỈ nội suy theo thời gian
    data_vars = [v for v in ds.data_vars if v != "region"]
    n_missing = int(sum(int(ds[v].isnull().sum()) for v in data_vars))
    if n_missing > 0:
        log.warning(f"Có {n_missing} giá trị missing. Đang nội suy theo thời gian...")
        ds = ds.interpolate_na(dim="time", method="linear", fill_value="extrapolate")

        # Nội suy có thể lấp cả ô ngoài ranh giới -> áp lại mask
        if mask is not None:
            ds = ds.where(mask.notnull())
            ds["region"] = mask.astype("float32")
            ds["region"].attrs["long_name"] = "region index (NaN = ngoài ranh giới)"

    # 8. Kiểm tra: ô ngoài ranh giới phải vẫn là NaN
    if mask is not None:
        if "tp" in ds:
            leak = int(ds["tp"].where(mask.isnull()).notnull().sum())
            if leak:
                log.error(f"Mask bị mất sau nội suy: {leak} giá trị ngoài ranh giới đã bị lấp.")
                sys.exit(1)
        n_outside = int(mask.isnull().sum())
        log.info(f"Đã giữ nguyên {n_outside} ô ngoài ranh giới là NaN.")

    # 9. Bỏ các coord kỹ thuật của ERA5 (không phải đặc trưng cho mô hình,
    #    và 'expver' là string -> sẽ làm XGBoost lỗi nếu lọt vào features)
    drop_coords = [c for c in ("number", "expver") if c in ds.coords]
    if drop_coords:
        if "expver" in ds.coords:
            ds.attrs["era5_expver"] = ",".join(sorted({str(v) for v in ds["expver"].values.ravel()}))
        ds = ds.drop_vars(drop_coords)
        log.info(f"Đã bỏ coord kỹ thuật: {drop_coords}")

    # 10. Lưu file
    out_file = cfg["paths"]["interim"] / f"era5_processed_{region_name}.nc"
    ds.to_netcdf(out_file)
    log.info(f"Đã lưu file sạch tại: {out_file}")

    return out_file


def _apply_region_override(cfg, region_key):
    """Đổi cfg['region'] sang preset trong region_presets."""
    presets = cfg.get("region_presets") or {}
    if region_key not in presets:
        log.error(f"Không có region '{region_key}' trong region_presets. Có: {list(presets)}")
        sys.exit(1)
    preset = dict(presets[region_key])
    preset.setdefault("name", region_key)
    cfg["region"] = preset
    log.info(f"Dùng region preset: {region_key} ({len(preset.get('provinces', []))} tỉnh)")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Tiền xử lý ERA5 (polygon clipping)")
    parser.add_argument("--config", default="config/config.yml")
    parser.add_argument("--region", default=None, help="Preset trong region_presets, vd: nam_trung_bo")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    if args.region:
        _apply_region_override(cfg, args.region)
    preprocess_era5(cfg)
    log.info("Hoàn tất tiền xử lý ERA5.")


if __name__ == "__main__":
    main()
