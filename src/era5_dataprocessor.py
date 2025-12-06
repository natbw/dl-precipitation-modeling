# src/era5_dataprocessor.py
# Handles preprocessing of zarr file to numpy arrays
# !pip install xarray zarr --quiet

from sklearn.preprocessing import StandardScaler
from pathlib import Path
import xarray as xr
import pandas as pd
import numpy as np
import joblib
import json

def era5_preprocessor(
    zarr_path,
    save_folder,
    train_range,
    val_range,
    test_range,
    features=None,
    target="total_precipitation",
    save_scaler=True
):
    """
    Preprocess ERA5/WeatherBench2 Zarr dataset to prepare inputs and outputs for
    machine learning models.
    
    Converts Zarr to pandas DataFrame, handles missing data, splits into train,
    validation, and test sets, scales features, and saves processed data as numpy arrays.
    
    Parameters
    ----------
    zarr_path : str or Path
        Path to the ERA5 Zarr dataset.
    save_folder : str or Path
        Folder to save preprocessed data and scaler.
    train_range : tuple of str
        Start and end dates for training data, e.g. ("1960-01-01", "2015-12-31").
    val_range : tuple of str
        Start and end dates for validation data.
    test_range : tuple of str
        Start and end dates for test data.
    features : list of str, optional
        List of features to use. If None, all variables except the target are used.
    target : str, optional
        Name of the target variable (default "total_precipitation").
    save_scaler : bool, optional
        Whether to save the fitted StandardScaler to disk (default True).
    
    Returns
    -------
    None
        Saves processed data to .npz and scaler to .pkl in save_folder.
    """
    # MAKE SURE FOLDER EXISTS
    save_folder = Path(save_folder)
    save_folder.mkdir(exist_ok=True)

    # LOAD ZARR FILE
    print(f"Loading Zarr file: {zarr_path}")
    ds = xr.open_zarr(zarr_path).load()

    if features is None:
        features = [v for v in ds.data_vars if v != target]

    print(f"Features: {features}")
    print(f"Target : {target}")

    # CONVERT ZARR TO DATAFRAME
    df = ds[features + [target]].to_dataframe()
    for col in ["latitude", "longitude"]:
        if col in df.columns:
            df = df.drop(columns=col)

    # DROP ANY FEATURE COLUMN WILL ALL NULL VALUES
    all_null_cols = [col for col in features if df[col].isnull().all()]
    if all_null_cols:
        print(f"Warning: The following feature columns are entirely null and will be dropped: {all_null_cols}")
        df = df.drop(columns=all_null_cols)
        features = [f for f in features if f not in all_null_cols]
    else:
        print("No empty feature columns found.")
        
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
    df = df.sort_index()

    # SAVE UNSCALED TO CSV
    df_to_save = df.copy()
    df_to_save.index = df_to_save.index.strftime("%m-%d-%Y")
    df_to_save.to_csv(save_folder / "era5_1960to2020_17feats_standard.csv")
    print(f"Unscaled dataframe saved to {save_folder / 'era5_1960to2020_17feats_standard.csv'}")

    # SPLIT DATA
    df_train = df[(df.index >= pd.to_datetime(train_range[0])) & (df.index <= pd.to_datetime(train_range[1]))]
    df_val = df[(df.index >= pd.to_datetime(val_range[0])) & (df.index <= pd.to_datetime(val_range[1]))]
    df_test = df[(df.index >= pd.to_datetime(test_range[0])) & (df.index <= pd.to_datetime(test_range[1]))]

    print(f"Train size: {df_train.shape}")
    print(f"Validation size: {df_val.shape}")
    print(f"Test size: {df_test.shape}")

    # SCALE FEATURES
    scaler = StandardScaler()
    scaler.fit(df_train[features])
    
    df_train_scaled = df_train.copy()
    df_val_scaled = df_val.copy()
    df_test_scaled = df_test.copy()
    
    df_train_scaled[features] = scaler.transform(df_train[features])
    df_val_scaled[features] = scaler.transform(df_val[features])
    df_test_scaled[features] = scaler.transform(df_test[features])
    
    # SAVE SCALER
    if save_scaler:
        scaler_path = save_folder / "era5_1960to2020_17feats_features_scaler.pkl"
        joblib.dump(scaler, scaler_path)
        print(f"Feature scaler saved to {scaler_path}")

    # SAVE AS ARRAYS
    dates_train = df_train_scaled.index.strftime("%m-%d-%Y").to_numpy()
    dates_val = df_val_scaled.index.strftime("%m-%d-%Y").to_numpy()
    dates_test = df_test_scaled.index.strftime("%m-%d-%Y").to_numpy()

    np.savez(
        save_folder / "era5_1960to2020_17feats_processed.npz",
        X_train=df_train_scaled[features].values,
        y_train=df_train[target].values,
        dates_train=dates_train,
    
        X_val=df_val_scaled[features].values,
        y_val=df_val[target].values,
        dates_val=dates_val,
    
        X_test=df_test_scaled[features].values,
        y_test=df_test[target].values,
        dates_test=dates_test,
    
        feature_names=np.array(features),
        target_name=target
    )
    
    print(f"Preprocessed data saved to {save_folder / 'era5_1960to2020_17feats_processed.npz'}")

# UNCOMMENT CODE TO RUN WITH CUSTOM DATA
# DATA USED FOR THIS PROJECT IS IN DATA FOLDER IN REPOSITORY ALREADY

# if __name__ == "__main__":
#     zarr_path = "../data/daily_era5_subset_1960-01-01_to_2020-12-31_17features.zarr"
#     save_folder = "../data"

#     train_range = ("1960-01-01", "2015-12-31")
#     val_range = ("2016-01-01", "2018-12-31")
#     test_range = ("2019-01-01", "2020-12-31")

#     features = None
#     target = "total_precipitation"
    
# era5_preprocessor(
#     zarr_path=zarr_path,
#     save_folder=save_folder,
#     train_range=train_range,
#     val_range=val_range,
#     test_range=test_range,
#     features=features,
#     target=target,
#     save_scaler=True
#     )