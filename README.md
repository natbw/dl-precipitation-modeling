# dl-precipitation-modeling


# Description

This project explores the use of machine learning and deep learning models on ERA5 reanalysis atmospheric data for precipitation prediction.

It provides scripts to preprocess ERA5 Zarr data, train baseline, LSTM, and Transformer models, and evaluate predictions including extreme precipitation events.


# Structure

* `data/`
    * `daily_era5_subset_1960-01-01_to_2020-12-31_17features.zarr` : ERA5 subset used for training and testing covering 60 years and 17 features.
    * `era5_subset_1980-01-01_to_2020-12-31.zarr`: ERA5 subset used for initial model implementation covering 40 years and 6 features.
    * Processed zarr files in `csv` or `.pkl` file types for model training.
* `notebooks/`
    * `01_era5_EDA.ipynb`: Jupyter notebook for exploring and visualizing ERA5 data.
    * `02_BaselinePipeline.ipynb`: Jupyter notebook for training and evaluating Baseline models.
    * `03_LSTMPipeline.ipynb`: Jupyter notebook for training and evaluating LSTM models.
    * `04_TransformerPipeline.ipynb`: Jupyter notebook for training and evaluating Transformer models.
* `src/`
    * `baseline_models.py`: Climatology, Persistence, and Linear Regression models.
    * `deeplearning_models.py`: LSTM model and helper functions.
    * `era5_dataloader.py`: Functions to load ERA5 data from Google Cloud Storage.
    * `era5_dataprocessor.py`: Preprocessing of Zarr files into NumPy arrays for training.
    * `evaluate.py`: Functions to compute RMSE, MAE, and extreme event metrics.
    * `train.py`: Scripts to train models.

# Requirements

The following dependencies need to be installed to run the code:

* Python 3.x
* PyTorch
* NumPy
* Pandas
* Matplotlib
* xarray
* netCDF4
* scikit-learn
* seaborn
* dask
* zarr
* fsspec

# Data

The data used for this project is stored in `data/` as a Zarr file:
`daily_era5_subset_1960-01-01_to_2020-12-31_17features.zarr`.

Contains 17 features including `total_precipitation`.

If you want to generate your own dataset, see the ERA5 documentation at https://cds.climate.copernicus.eu/datasets

# Preprocessing ERA5 Data

Use `era5_dataprocessor.py` to convert Zarr files into NumPy arrays for training:

```
from src.era5_dataprocessor import era5_preprocessor
from pathlib import Path

zarr_path = "data/daily_era5_subset_1960-01-01_to_2020-12-31_17features.zarr"

save_folder = Path("data/processed")

train_range = ("1960-01-01", "2015-12-31")
val_range   = ("2016-01-01", "2018-12-31")
test_range  = ("2019-01-01", "2020-12-31")

era5_preprocessor(
    zarr_path=zarr_path,
    save_folder=save_folder,
    train_range=train_range,
    val_range=val_range,
    test_range=test_range,
    target="total_precipitation",
    save_scaler=True
)
```

This generates:
- `era5_1960to2020_17feats_processed.npz` processed data arrays
- `era5_1960to2020_17feats_features_scaler.pkl` feature scalar

# Training Models

To train and explore different models, run the following notebooks:

- `02_BaselinePipeline.ipynb` for training baseline models
- `03_LSTMPipeline.ipynb` for training LSTM models
- `04_TransformerPipeline.ipynb` for training Transformer models

# Running in Google Colab

1. Mount Google Drive to access data:

```
from google.colab import drive
drive.mount('/content/drive')
```

2. Install dependencies in Colab:

`!pip install numpy pandas matplotlib xarray netCDF4 scikit-learn seaborn dask zarr fsspec torch --quiet`

3. Update paths to point to Drive folder:

`data_path = "/content/drive/MyDrive/ERA5/processed/era5_1960to2020_17feats_processed.npz"`