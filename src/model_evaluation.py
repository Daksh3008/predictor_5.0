# src/model_evaluation.py

import pandas as pd




# -----------------------------------------------------------
# SAVE BACKTEST OUTPUT TO EXCEL (5 sheets + summary)
# -----------------------------------------------------------
def save_backtest_excel(out_path, results, dates, actual_prices):

    df_lstm = pd.DataFrame({
        "Date": dates,
        "Actual": actual_prices,
        "Predicted": results["lstm"],
        "% Diff": ((results["lstm"] - actual_prices) / actual_prices) * 100,
    })

    df_tcn = pd.DataFrame({
        "Date": dates,
        "Actual": actual_prices,
        "Predicted": results["tcn"],
        "% Diff": ((results["tcn"] - actual_prices) / actual_prices) * 100,
    })

    df_lgbm = pd.DataFrame({
        "Date": dates,
        "Actual": actual_prices,
        "Predicted": results["lgbm"],
        "% Diff": ((results["lgbm"] - actual_prices) / actual_prices) * 100,
    })


    df_hmm = pd.DataFrame({
        "Date": dates,
        "Regime": results["hmm"]
    })

    df_ens = pd.DataFrame({
        "Date": dates,
        "Actual": actual_prices,
        "Predicted": results["ensemble"],
        "% Diff": ((results["ensemble"] - actual_prices) / actual_prices) * 100,
    })

    df_summary = pd.DataFrame({
        "Model": ["LSTM", "TCN", "LightGBM", "Ensemble"],
        "RMSE": [
            results["rmse_lstm"],
            results["rmse_tcn"],
            results["rmse_lgbm"],
            results["rmse_ensemble"],
        ],
    })

    # Use clean context manager (no deprecated save())
    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        df_lstm.to_excel(writer, sheet_name="LSTM_Attn", index=False)
        df_tcn.to_excel(writer, sheet_name="TCN", index=False)
        df_lgbm.to_excel(writer, sheet_name="LightGBM", index=False)
        df_hmm.to_excel(writer, sheet_name="HMM_Regimes", index=False)
        df_ens.to_excel(writer, sheet_name="Ensemble", index=False)
        df_summary.to_excel(writer, sheet_name="Summary", index=False)

    print(f"✔ Excel saved: {out_path}")
