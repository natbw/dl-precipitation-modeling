# src/era5_dataprocessor.py
# Handles preprocessing of zarr file to numpy arrays
# !pip install xarray zarr --quiet

from sklearn.preprocessing import StandardScaler
from pathlib import Path
import xarray as xr
import pandas as pd
import numpy as np

def era5_preprocessor(
    zarr_path,
    save_folder,
    train_range,
    val_range,
    test_range,
    features=None,
    target="total_precipitation"
):
    # MAKE SURE FOLDER EXISTS
    save_folder = Path(save_folder)
    save_folder.mkdir(exist_ok=True)

    # LOAD ZARR FILE
    print(f"Loading Zarr file from {zarr_path} ...")
    ds = xr.open_zarr(zarr_path).load()

    # FILTER FEATURES IF SPECIFIED
    if features is None:
        features = [v for v in ds.data_vars if v != target]

    print(f"Selected features: {features}")
    print(f"Target: {target}")

    # CONVERT ZARR TO DATAFRAME
    df = ds[features + [target]].to_dataframe()
    df = df.drop(columns=[col for col in ["latitude", "longitude"] if col in df.columns])

    # CONFIRM NO NULL VALUES
    if df.isnull().values.any():
        missing_count = df.isnull().sum().sum()
        raise ValueError(f"Dataset contains {missing_count} missing values. Please handle before proceeding.")
    else:
        print("No missing values found.")

    # REORDER COLUMNS AND SET DATE TO INDEX
    cols = [c for c in df.columns if c != target] + [target]
    df = df[cols]
    df.index = pd.to_datetime(df.index)

    # SAVE UNSCALED TO CSV
    df_to_save = df.copy()
    df_to_save.index = df_to_save.index.strftime("%m-%d-%Y")
    df_to_save.to_csv(save_folder / "era5_standard.csv")
    print(f"Unscaled dataframe saved to {save_folder / 'era5_standard.csv'}")

    # SCALE FEATURES
    scaler = StandardScaler()
    feature_values = df[features].values
    scaled_features = scaler.fit_transform(feature_values)
    df_scaled = pd.DataFrame(scaled_features, columns=features, index=df.index)
    df_scaled[target] = df[target].values

    # SPLIT DATA
    train_mask = (df_scaled.index >= pd.to_datetime(train_range[0])) & (df_scaled.index <= pd.to_datetime(train_range[1]))
    val_mask = (df_scaled.index >= pd.to_datetime(val_range[0])) & (df_scaled.index <= pd.to_datetime(val_range[1]))
    test_mask = (df_scaled.index >= pd.to_datetime(test_range[0])) & (df_scaled.index <= pd.to_datetime(test_range[1]))

    df_train = df_scaled.loc[train_mask]
    df_val = df_scaled.loc[val_mask]
    df_test = df_scaled.loc[test_mask]

    dates_train = df_train.index.strftime("%m-%d-%Y").to_numpy()
    dates_val = df_val.index.strftime("%m-%d-%Y").to_numpy()
    dates_test = df_test.index.strftime("%m-%d-%Y").to_numpy()

    print(f"Train size: {df_train.shape}")
    print(f"Validation size: {df_val.shape}")
    print(f"Test size: {df_test.shape}")

    # SAVE AS ARRAYS
    np.savez(
        save_folder / "era5_processed.npz",
        X_train=df_train[features].values,
        y_train=df_train[target].values,
        dates_train=dates_train,
        X_val=df_val[features].values,
        y_val=df_val[target].values,
        dates_val=dates_val,
        X_test=df_test[features].values,
        y_test=df_test[target].values,
        dates_test=dates_test,
        feature_names=np.array(features),
        target_name=target
    )

    print(f"Preprocessed data saved to {save_folder / 'era5_processed.npz'}")

# UNCOMMENT CODE TO RUN WITH CUSTOM DATA
# DATA USED FOR THIS PROJECT IS IN DATA FOLDER IN REPOSITORY ALREADY

# if __name__ == "__main__":
#     zarr_path = "../data/era5_subset_1980-01-01_to_2020-12-31.zarr"
#     save_folder = "../data"

#     train_range = ("1980-01-01", "2015-12-31")
#     val_range = ("2016-01-01", "2018-12-31")
#     test_range = ("2019-01-01", "2020-12-31")

#     features = None
#     target = "total_precipitation"
    
# era5_preprocessor(
#         zarr_path=zarr_path,
#         save_folder=save_folder,
#         train_range=train_range,
#         val_range=val_range,
#         test_range=test_range,
#         features=features,
#         target=target
#     )