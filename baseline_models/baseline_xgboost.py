import json
import numpy as np
from datetime import datetime
from logger import logger
from train.baseline_dataloader import load_baseline_data


def train_xgboost(cfg_path='config.json'):
    cfg = json.load(open(cfg_path))
    data_cfg = cfg['data']

    logger.info("=== XGBoost Baseline ===")

    X_train, X_val, _, _, y_train, y_val = load_baseline_data(
        data_cfg['feature_path'],
        data_cfg['label_path'],
        test_size=data_cfg['test_size'],
        random_state=data_cfg['random_state'],
        use_last_timestep=True,
        return_train=True,
    )
    logger.info(f"X_train: {X_train.shape}, X_val: {X_val.shape}")

    import xgboost as xgb

    model = xgb.XGBRegressor(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        early_stopping_rounds=30,
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )

    preds_val = model.predict(X_val)

    # Compute metrics
    signed = preds_val - y_val
    sigma = float(np.std(signed))
    log10_mae = float(np.mean(np.abs(signed)))
    mean_bias = float(np.mean(signed))
    within_1sigma = float(np.mean(np.abs(signed) < sigma) * 100.0)
    within_2sigma = float(np.mean(np.abs(signed) < 2 * sigma) * 100.0)

    logger.info(f"  Mean bias (μ)  : {mean_bias:+.4f} orders")
    logger.info(f"  1σ             : {sigma:.4f} orders")
    logger.info(f"  Log10-MAE      : {log10_mae:.4f} orders")
    logger.info(f"  Within 1σ      : {within_1sigma:.1f}%")
    logger.info(f"  Within 2σ      : {within_2sigma:.1f}%")

    # Save model
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    import pickle
    with open(f'params/xgb_baseline_{timestamp}.pkl', 'wb') as f:
        pickle.dump(model, f)
    logger.info(f"Model saved to params/xgb_baseline_{timestamp}.pkl")

    return {
        'log10_mae': log10_mae,
        'sigma': sigma,
        'mean_bias': mean_bias,
        'within_1sigma': within_1sigma,
        'within_2sigma': within_2sigma,
        'preds': preds_val,
        'gts': y_val,
    }


if __name__ == "__main__":
    from logger import setup_file_handler
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    setup_file_handler(f"xgb_{ts}")
    train_xgboost()
