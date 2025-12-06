# src/era5_dataloader.py
# Handles loading ERA5/WeatherBench2 data from Google Cloud Storage
# !pip install xarray gcsfs zarr numcodecs --quiet

from dask.diagnostics import ProgressBar
from pathlib import Path
import xarray as xr
import numcodecs
import gcsfs
import os

def era5_data_loader(
        base_url,
        variables,
        region,
        levels=None,
        average_levels=True,
        daily_resample=True,
        time_range=None,
        data_folder="./test_data",
        batch_years=5
):
    """
    Load ERA5/WeatherBench2 data from Google Cloud Storage, subset spatially and
    temporally, resample daily, optionally average over pressure levels, and save
    as a local Zarr dataset.

    Parameters
    ----------
    base_url : str
        Google Cloud Storage URL pointing to the ERA5/WeatherBench2 Zarr dataset.
    variables : list of str
        List of variable names to load from the dataset.
    region : dict
        Spatial subset parameters. Must include 'type':
            - "point" -> {'type': 'point', 'lat': float, 'lon': float}
            - "box" -> {'type': 'box', 'lat_min': float, 'lat_max': float, 
                        'lon_min': float, 'lon_max': float}
    levels : list, optional
        List of pressure levels to select (default is None, no level selection)
    average_levels : bool, optional
        Whether to average variables over pressure levels if present (default True)
    daily_resample : bool, optional
        Whether to resample data to daily resolution (default True)
    time_range : tuple, optional
        Temporal subset in the form ('YYYY-MM-DD', 'YYYY-MM-DD'). If None, uses all time steps.
    data_folder : str or Path, optional
        Local folder where the Zarr file will be saved (default './test_data')
    batch_years : int, optional
        Number of years per batch when saving yearly chunks (default 5). Currently unused.

    Returns
    -------
    list of Path
        List containing the path to the saved Zarr dataset.

    Notes
    -----
    - Daily resampling converts 6-hourly precipitation to daily total precipitation (in mm).
    - Non-precipitation variables are averaged over the day.
    - Large datasets may take hours to download and save.
    """
    print("\n=== ERA5 DATA LOADER ===\n")

    # CONNECT TO GOOGLE CLOUD STORAGE
    fs = gcsfs.GCSFileSystem(token="anon")
    mapper = fs.get_mapper(base_url)
    ds_full = xr.open_zarr(mapper, consolidated=True)
    available_vars = list(ds_full.data_vars)
    print(f"Available vars: {sorted(available_vars)}\n")

    # VALIDATE VARIABLES
    valid_vars = []
    for v in variables:
        if v in available_vars:
            valid_vars.append(v)
        else:
            print(f"WARNING: Variable '{v}' not found in WB2 store, skipping.")

    if not valid_vars:
        raise ValueError("None of the requested variables exist in the dataset.")

    print(f"Loading variables: {valid_vars}\n")

    # SPATIAL SUBSET
    print("Performing spatial subset...\n")
    ds = ds_full[valid_vars]

    if region["type"] == "point":
        ds = ds.sel(
            latitude=region["lat"],
            longitude=region["lon"],
            method="nearest"
        )
    elif region["type"] == "box":
        ds = ds.sel(
            latitude=slice(region["lat_max"], region["lat_min"]),
            longitude=slice(region["lon_min"], region["lon_max"])
        )
    else:
        raise ValueError("region['type'] must be 'point' or 'box'")

    # PRESSURE LEVEL SUBSET
    if levels is not None:
        if "level" not in ds.dims:
            raise ValueError("Selected variables do not contain a 'level' dimension.")

        print(f"Selecting pressure levels: {levels}")
        ds = ds.sel(level=levels)
        print("Pressure level subset complete.\n")

    # TEMPORAL SUBSET
    print("Applying temporal subset...\n")
    if time_range is not None:
        ds = ds.sel(time=slice(time_range[0], time_range[1]))
        time_name = f"subset_{time_range[0]}_to_{time_range[1]}"
        print(f"Time coverage: {ds.time.values[0]} - {ds.time.values[-1]}\n")
    else:
        time_name = "alltime"

    # DAILY RESAMPLING
    if daily_resample:
            print("Applying daily resampling...\n")

            precip_vars = [v for v in ds.data_vars if "precip" in v.lower()]
            non_precip_vars = [v for v in ds.data_vars if v not in precip_vars]

            ds_daily = xr.Dataset()
            
            # Average non-precipitation variables
            if non_precip_vars:
                ds_daily = ds[non_precip_vars].resample(time="1D").mean()
            
            # Sum precipitation variables to daily total, clip negatives, convert to mm
            for pv in precip_vars:
                daily_sum = ds[pv].resample(time="1D").sum()
                daily_sum = daily_sum.clip(min=0)
                daily_sum = daily_sum * 1000.00

                resamp_name = "total_precipitation"
                ds_daily[resamp_name] = daily_sum
                ds_daily[resamp_name].attrs["units"] = "mm"

            ds = ds_daily
            print("Daily aggregation complete.\n")

    # LEVEL AVERAGING
    if average_levels and "level" in ds.dims:
        print("Averaging over pressure levels...\n")
        for v in ds.data_vars:
            if "level" in ds[v].dims:
                ds[v] = ds[v].mean(dim="level")
        ds = ds.drop_vars("level")
        print("Pressure-level averaging complete.\n")

    # MAKE OUTPUT FOLDER
    os.makedirs(data_folder, exist_ok=True)
    data_folder = Path(data_folder)

    # SAVE SINGLE FILE
    print("Saving dataset...")

    filename = f"daily_era5_{time_name}_{len(valid_vars)}features.zarr"
    output_path = data_folder / filename

    # Chunk the dataset for faster writing and compression
    ds = ds.chunk({"time": 5000})
    encoding = {v: {"compressor": numcodecs.Blosc()} for v in ds.data_vars}
    with ProgressBar():
        ds.to_zarr(output_path, mode="w", encoding=encoding, zarr_format=2)
    print("\n=== ERA5 downloader complete ===")
    return [output_path]

    # # SAVE IN YEARS
    # chunk_paths = []
    # start_year = int(str(ds.time.values[0])[:4])
    # end_year = int(str(ds.time.values[-1])[:4])
    # for y_start in range(start_year, end_year + 1, batch_years):
    #     y_end = min(y_start + batch_years - 1, end_year)
    #     chunk_ds = ds.sel(time=slice(f"{y_start}-01-01", f"{y_end}-12-31"))
    #     if len(chunk_ds.time) == 0:
    #         continue

    #     filename = f"era5_{y_start}_{y_end}.zarr"
    #     local_path = data_folder / filename
    #     print(f"Saving {y_start}-{y_end} to {local_path} ...")

    #     encoding = {v: {"compressor": numcodecs.Blosc()} for v in chunk_ds.data_vars}
    #     chunk_ds = chunk_ds.chunk({"time": 1000})
    #     with ProgressBar():
    #         chunk_ds.to_zarr(local_path, mode="w", encoding=encoding, zarr_format=2)

    #     chunk_paths.append(local_path)

    # print("\n=== ERA5 loader complete ===")
    # return chunk_paths


# UNCOMMENT THE CODE BELOW TO RUN. DEPENDING ON DATE RANGE AND VARIABLES SELECTED
# THIS CAN TAKE HOURS TO RUN AND COULD BE TOO LARGE OF A FILE TO DOWNLOAD
# USE THE LOCAL COPY ALREADY DOWNLOADED TO ../data IN THIS REPOSITORY
    
# if __name__ == "__main__":

#     # DATA LOADER PARAMETERS
#     base_url = "gs://weatherbench2/datasets/era5/1959-2023_01_10-6h-240x121_equiangular_with_poles_conservative.zarr"

#     selected_variables = [
#         "specific_humidity",
#         "vertical_velocity",
#         "temperature",
#         "relative_humidity",
#         "geopotential",
#         "u_component_of_wind",
#         "10m_u_component_of_wind",
#         "v_component_of_wind",
#         "10m_v_component_of_wind",
#         "wind_speed",
#         "divergence",
#         "total_column_water_vapour",
#         "total_cloud_cover",
#         "mean_sea_level_pressure",
#         "2m_temperature",
#         "sea_surface_temperature",
#         "total_precipitation_6hr"
#     ]

#     region = {"type": "point", "lat": 30.0, "lon": 270.0}

#     selected_levels = None
#     average_levels = True
#     daily_resample = True

#     time_range = ("1960-01-01", "2020-12-31")

#     data_folder = Path("..") / "test_data"

#     local_file = era5_data_loader(
#     base_url=base_url,
#     variables=selected_variables,
#     region=region,
#     levels=selected_levels,
#     average_levels=average_levels,
#     daily_resample=daily_resample,
#     time_range=time_range,
#     data_folder=data_folder,
#     batch_years=5
# )