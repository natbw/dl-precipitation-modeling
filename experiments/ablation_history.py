# experiments/ablation_history.py
# Run ablation study on effects of history window

import json
import os
from pathlib import Path
from src.train import train_baseline_models, train_lstm_model, train_transformer_model

def run_ablation_history(
    data_path,
    T_values=[1, 3, 5, 7, 14, 30],
    horizon=3,
    lstm_epochs=40,
    transformer_epochs=40,
    device="cuda",
    save_results=True,
    save_curves=True,
    output_dir="ablation_outputs"
):

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {}

    for T in T_values:
        print("\n" + "="*70)
        print(f"Running ablation for history window T = {T}")
        print("="*70)

        # BASELINE MODELS
        baseline_res = train_baseline_models(
            data_path=data_path,
            T=T,
            horizon=horizon,
            p_window=T
        )

        # LSTM MODEL
        lstm_res = train_lstm_model(
            data_path=data_path,
            T=T,
            horizon=horizon,
            epochs=lstm_epochs,
            device=device
        )

        # TRANSFORMER MODEL
        transformer_res = train_transformer_model(
            data_path=data_path,
            T=T,
            horizon=horizon,
            epochs=transformer_epochs,
            device=device
        )

        # COMPILE METRICS
        results[T] = {
            "baseline": {
                model_name: {
                    "rmse": baseline_res[model_name]["rmse"],
                    "mae": baseline_res[model_name]["mae"]
                }
                for model_name in baseline_res
            },
            "lstm": {
                "rmse": lstm_res["LSTM"]["rmse"],
                "mae": lstm_res["LSTM"]["mae"],
                "train_loss": lstm_res["LSTM"]["train_loss"],
                "val_loss": lstm_res["LSTM"]["val_loss"]
            },
            "transformer": {
                "rmse": transformer_res["transformer"]["rmse"],
                "mae": transformer_res["transformer"]["mae"],
                "train_loss": transformer_res["transformer"]["train_loss"],
                "val_loss": transformer_res["transformer"]["val_loss"]
            }
        }

        # SAVE TRAINING INFO
        if save_curves:
            curve_path = output_dir / f"training_curves_T{T}.json"
            json.dump({
                "lstm_train": lstm_res["LSTM"]["train_loss"],
                "lstm_val": lstm_res["LSTM"]["val_loss"],
                "transformer_train": transformer_res["transformer"]["train_loss"],
                "transformer_val": transformer_res["transformer"]["val_loss"]
            }, open(curve_path, "w"), indent=4)

    # SAVE
    if save_results:
        out_file = output_dir / "ablation_history_results.json"
        json.dump(results, open(out_file, "w"), indent=4)
        print("\nAblation study complete. Results saved to:", out_file)

    return results
