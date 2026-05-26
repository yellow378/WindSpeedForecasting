# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a fork of TSlib (Time Series Library, THUML/Tsinghua) customized for **wind speed forecasting** with spatial (graph) modeling. It provides a unified framework for training and evaluating deep time series models across five tasks: long-term forecasting, short-term forecasting, imputation, anomaly detection, and classification. The fork adds wind-specific models (LightTime, GAT), spatial forecasting, and analysis tooling.

## Running Experiments

**Train a single model (long-term forecasting):**
```bash
python run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/processed \
  --data_path Turb1.csv \
  --model_id Wind_LightTime \
  --model LightTime \
  --data custom \
  --features MS --target Wspd \
  --seq_len 72 --label_len 18 --pred_len 36 \
  --enc_in 10 --dec_in 10 --c_out 1 \
  --d_model 512 --n_heads 8 --e_layers 2 \
  --patch_len 36 --stride 18 \
  --train_epochs 40 --patience 10 --batch_size 256 --learning_rate 0.001 \
  --use_gpu True
```

**Test only (inference):** Set `--is_training 0` with the same args.

**Spatial forecasting (GAT):**
```bash
sh ./scripts/spatial/Light-GAT.sh
```

**LightST (spatio-temporal, 3-stage training):**
```bash
sh ./scripts/spatial/LightST.sh
```

**Batch hyperparameter search for all WindSpeed models:**
```bash
sh ./run.sh
```

Individual model scripts are in `scripts/long_term_forecast/WindSpeed_script/`. Each performs a grid search over `seq_len`, `e_layers`, `d_model`, `n_heads`, `patch_len` etc., logging to `logs/LongForecasting/`.

**Key CLI flags specific to this fork:**
- `--task_name spatial` — routes to `Exp_Spatial_Long_Term_Forecast`
- `--task_name lightst` — routes to `Exp_LightST` (LightST with 3-stage training)
- `--decomposition 1` — enables trend/seasonal decomposition in LightTime
- `--edge_file` — path to `.npy` edge index file for GAT
- `--debug` — verbose logging
- `--inverse` — inverse-transform predictions before metric computation
- `--n_clusters` — number of latent clusters for the hierarchical encoder (default: 8)
- `--gnn_layers` — number of E-MGAT layers (default: 2)
- `--stage1_epochs/stage2_epochs/stage3_epochs` — per-stage epochs for LightST
- `--static_graph_dir` — directory with pre-computed 16-sector static graphs

## Architecture

### Entry Point and Dispatch

`run.py` parses CLI args, selects an `Exp_*` class based on `--task_name`, then calls `exp.train(setting)` and `exp.test(setting)`.

```
run.py
  └── exp/exp_basic.py          # Base class, model_dict registry, device setup
      ├── exp_long_term_forecasting.py    # Default forecasting (encoder-decoder or direct)
      ├── exp_spatial_long_term_forecasting.py  # Graph-based spatial forecasting (GAT)
      ├── exp_lightst.py                   # LightST 3-stage progressive training
      ├── exp_short_term_forecasting.py   # M4 competition
      ├── exp_imputation.py
      ├── exp_anomaly_detection.py
      └── exp_classification.py
```

### Model Registry and Convention

Models are registered in `exp/exp_basic.py` `model_dict`. To add a new model:
1. Create `models/YourModel.py` with a `Model(nn.Module)` class taking a single `configs` (argparse namespace) argument
2. Import it in `exp/exp_basic.py` and add to `model_dict`

**Forward signatures differ by model type:**
- Encoder-decoder models (Transformer, Autoformer, Informer, etc.): `forward(x_enc, x_mark_enc, x_dec, x_mark_dec)`
- Direct models (PatchTST, SegRNN, SparseTSF, LightTime, DLinear): `forward(x)`
- GAT: `forward(data, device)` where `data` is a PyTorch Geometric `Batch` object
- LightST: `forward(data, device)` with internal graph management — selects sector based on wind direction, computes dynamic edge features, runs E-MGAT. Also has `forward_stage1(x)` for temporal-only training.

The experiment classes handle this: `Exp_Long_Term_Forecast` checks model name to decide whether to pass decoder inputs; `Exp_Spatial_Long_Term_Forecast` uses the graph data loader.

### Data Pipeline

`data_provider/data_factory.py` routes `--data` to a dataset class:
- `custom` → `Dataset_Custom` — generic CSV loader (used for wind data). Train/val/test split: 70/10/20.
- `spatial_wind` → uses `data_provider/gat_data_loader.py` — loads multiple turbine CSVs, builds PyG `Data` objects with `edge_index` and `edge_attr`.
- Built-in: `ETTh1/ETTh2`, `ETTm1/ETTm2`, `PSM`, `MSL`, `SMAP`, `SMD`, etc.

Standard forecasting returns 4-tuples: `(seq_x, seq_y, seq_x_mark, seq_y_mark)` where shapes are `[B, seq_len, C]`, `[B, label_len+pred_len, C]`, time marks.

### Static Graph Pre-computation

Before running LightST, pre-compute the 16-sector static graphs offline:
```bash
python utils/static_graph_builder.py \
  --data_dir ./dataset/spatial_wind \
  --positions_file ./dataset/spatial_wind/positions.npy \
  --output_dir ./dataset/static_graphs
```
This creates per-sector `edge_index`, `adj_weights`, and 11-dim `static_edge_attr` files.

### Key Fork-Specific Models

- **LightTime** (`models/LightTime.py`): Lightweight model with optional decomposition, patching, and channel-independent processing. Uses `--decomposition`, `--patch_len`, `--stride`.
- **GAT** (`models/GAT.py`): Combines LightTime temporal encoder with Graph Attention Network spatial aggregation. Requires `--task_name spatial` and `--edge_file`.
- **LightST** (`models/LightST.py`): Spatial-temporal model with dynamic sparse graph attention for multi-turbine wind speed forecasting. Key components: hierarchical encoder (shared + cluster + individual adapters), 16-sector wind direction conditioned static graphs, 25-dim edge features (11 static + 14 dynamic), edge-enhanced multi-head GAT (E-MGAT), and three-stage progressive training. Uses `--task_name lightst`.
- **SegRNN**, **SparseTSF**, **DLinearSingle**: Additional baseline models added for comparison.

### Graph Construction Pipeline

`analysis/graph/` contains the full pipeline for building spatial graphs:
1. `0_process.ipynb` — data preprocessing
2. `1_correlation/` — wind speed/direction correlation matrices
3. `2_cluster/` — node clustering from turbine locations
4. `3_static_graph/` — static graph with edge features (KNN, fully connected)
5. `5_visualize/` — graph visualization

`analysis/graph2/1_sector/` — sector-based (wind direction partition) correlation analysis.

### Layers and Utilities

- `layers/` — shared building blocks (embeddings, attention mechanisms, encoder/decoder pairs). Each model-specific architecture (Autoformer, Crossformer, ETSformer, etc.) has its own `*_EncDec.py` file.
- `utils/metrics.py` — MAE, MSE, RMSE, MAPE, R2, CORR
- `utils/tools.py` — EarlyStopping, StandardScaler, learning rate scheduling, visualization
- `utils/losses.py` — MAPE, SMAPE, MASE losses (short-term forecasting)
- `utils/soft_dtw.py` — Soft Dynamic Time Warping loss (alternative loss)

## Conventions

- Random seed is fixed at 496496 in `run.py`
- Wind speed experiments use `--features MS` (multivariate-to-single), `--target Wspd`, `--enc_in 10`, `--dec_in 10`, `--c_out 1`
- `--label_len` is typically set to 18 for wind speed experiments
- Logs go to `logs/LongForecasting/`, checkpoints to `checkpoints/`, results to `results/`
- No formal test suite — validation is done by running experiments and checking metrics in logs
