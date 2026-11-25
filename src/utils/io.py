import xarray as xr

def open_netcdf(path, bbox=None):
    ds = xr.open_dataset(path)
    if bbox:
        lon_min, lat_min, lon_max, lat_max = bbox
        ds = ds.sel(lon=slice(lon_min, lon_max),
                    lat=slice(lat_max, lat_min))  # lat giảm dần trong ERA5
    return ds