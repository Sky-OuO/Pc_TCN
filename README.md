# Satellite Collision Probability Prediction using TCN
# Abstract
This project proposes a hybrid temporal convolutional network (TCB) based algorithm to predict the probability of collision between two space objects, using only the two-line elements information and time of closest approach. It aims to efficiently predict the probability of collision by cross-correlating satellite’s time-series positioning data, while handling the highly imbalanced samples.

Here, the dilated TCN blocks allow the algorithm to capture the multi-scale temporal patterns in geometric sequences. In addition, a logarithmic output domain is introduced to address the wide dynamic range of probability of collision, thus, improves the algorithm’s learning stability. Further, an imbalance-aware sampling strategy is introduced to enhance the exposure of rare high-risk cases during training processes. 

## Overview

```
┌─────────────────────┐     ┌─────────────────────┐     ┌─────────────────────┐
│  Data Collection     │ ──▶ │  Feature Generation  │ ──▶ │  TCN Training        │
│  (CelesTrak API)     │     │  (SGP4 Propagation)  │     │  (Regression)        │
└─────────────────────┘     └─────────────────────┘     └─────────────────────┘
```

## Project Structure

```
EE5003/
├── data_collection_pipline.py   # Adaptive data collection from CelesTrak
├── feature_generator.py         # feature extraction via SGP4
├── TCN_train.py                 # TCN model definition, training & evaluation
├── features.npy                 # Generated feature arrays
├── labels.npy                   # Ground-truth collision probabilities
└── data/
    ├── collected_tle_datas.json  # Collected TLE data with collision events
```

## Pipeline

### 1. Data Collection (`data_collection_pipline.py`)

An adaptive snowball-sampling collector that gathers satellite conjunction events from the CelesTrak SOCRATES database:

- **Seed Initialization**: Fetches the top high-risk conjunction events as initial seeds.
- **Iterative Expansion**: Queries each satellite's conjunction history and discovers new satellites.
- **Distribution Balancing**: Monitors the collision probability distribution across predefined bins ($[10^{-7}, 10^{-6})$, $[10^{-6}, 10^{-5})$, ..., $[10^{-2}, 1.0)$) and prioritizes under-represented ranges.
- **TLE Fetching**: After collection, retrieves the latest TLE data for every involved satellite from CelesTrak's GP API.
- **Output Format**: JSON array of events, each containing TCA timestamp, ground-truth $P_c$, and two-line TLE sets for both satellites.

```bash
python data_collection_pipline.py
```

### 2. Feature Generation (`feature_generator.py`)

Generates time-series orbital features from TLE data using SGP4 propagation:

- Directly converts each collected event into one feature sequence without synthetic augmentation.
- Propagates both satellites backward from TCA (Time of Closest Approach) over a configurable time window.
- Computes per-timestep features at each point:
  - **Relative dynamics**: distance, closure rate, relative speed, radial/tangential velocity components, relative position & velocity vectors
  - **Absolute state**: position magnitude, velocity magnitude, position & velocity vectors of satellite 1
  - **Temporal**: time-to-TCA
  - **TLE age**: days since epoch for both satellites (data quality indicators)

```bash
python feature_generator.py
```

Output: `features.npy` (shape: `[N, T, F]`) and `labels.npy` (shape: `[N]`), where N = number of events, T = time steps, F = feature dimensions.

### 3. Model Training (`TCN_train.py`)

Trains a Temporal Convolutional Network to predict $\log_{10}(P_c)$:

**Architecture:**
- **TCN backbone** with dilated causal convolutions (5 layers, 32 channels each) for processing geometric/dynamic time-series features
- **MLP branch** for TLE age features (2D → 16 → 8)
- **Feature fusion**: concatenation of TCN output and age MLP output
- **Regression head**: FC layers outputting $\log_{10}(P_c)$

**Training Details:**
- **Loss**: Decade-Scaled MSE — weights errors by order of magnitude to handle the wide dynamic range of $P_c$ values ($10^{-7}$ to $1.0$)
- **Sampling**: Weighted random sampler to oversample high-risk events ($P_c > 10^{-4}$)
- **Optimizer**: AdamW with learning rate $5 \times 10^{-5}$ and weight decay $10^{-5}$
- **Early stopping**: patience of 30 epochs
- **Sequence handling**: zero-padding/truncation to fixed length (600 timesteps)

### Installation

```bash
pip install -r requirements.txt
```

### Full Pipeline

```bash
# Step 1: Collect conjunction events and TLE data(periodically!)
python data_collection_pipline.py

# Step 2: Generate features from TLE data
python feature_generator.py

# Step 3: Train the TCN model
python TCN_train.py
```

## Data Sources

- **[CelesTrak SOCRATES](https://celestrak.org/SOCRATES/)** — Satellite Orbital Conjunction Reports Assessing Threatening Encounters in Space. Provides conjunction event data including TCA, miss distance, relative velocity, and collision probability.
- **[CelesTrak GP API](https://celestrak.org/NORAD/elements/)** — General Perturbations orbital data in TLE format for individual satellites by NORAD catalog number.

