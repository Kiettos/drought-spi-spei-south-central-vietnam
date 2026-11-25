import cdsapi

# DÁN TRỰC TIẾP KEY CỦA BẠN VÀO ĐÂY
MY_KEY = "url: https://cds.climate.copernicus.eu/api\nkey: c8686287-3610-4ce2-aeab-6e34711d4aa0"

# Khởi tạo client với key trực tiếp
c = cdsapi.Client(url="https://cds.climate.copernicus.eu/api", key="c8686287-3610-4ce2-aeab-6e34711d4aa0")

print("Đang thử tải 1 tháng dữ liệu...")
try:
    c.retrieve(
        'reanalysis-era5-single-levels-monthly-means',
        {
            'product_type': 'monthly_averaged_reanalysis',
            'variable': '2m_temperature',
            'year': '2023',
            'month': '01',
            'time': '00:00',
            'area': [16.5, 102, 8.5, 110],
            'format': 'netcdf',
        },
        'test_download.nc')
    print("✅ THÀNH CÔNG! Key của bạn hoàn toàn hợp lệ.")
except Exception as e:
    print(f"❌ VẪN LỖI: {e}")