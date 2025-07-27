G-GraphTST is a novel Global-Embedding Graph-Encoder Time Series Transformer designed for multivariate long-term photovoltaic power forecasting. Our model addresses the limitations of existing methods by effectively modeling spatiotemporal correlations at different perceptual levels and incorporating interactive learning between spatial and temporal dependencies.
🌟 Key Features

Global Embedding Mechanism: Captures periodicity and trends of the entire multi-node photovoltaic system
Dynamic Graph Construction: Real-time adaptive generation of adjacency matrices based on node embedding similarity
Spatiotemporal Fusion: Interactive learning framework combining spatial and temporal modules
Superior Performance: Outperforms state-of-the-art methods on multiple benchmark datasets
Strong Generalizability: Applicable to various time series forecasting domains beyond photovoltaic systems

🚀 Quick Start
Installation
bash# Clone the repository
git clone https://github.com/username/G_GraphTST.git
cd G_GraphTST

# Create virtual environment
conda create -n g_graphtst python=3.8
conda activate g_graphtst

# Install dependencies
pip install -r requirements.txt
Basic Usage
bash# Run G-GraphTST on Solar dataset
python run.py --model GraphPatchTST --data solar --seq_len 96 --pred_len 96 --batch_size 16

# Run on ETT dataset with custom parameters
python run.py --model GraphPatchTST --data ETTh1 --seq_len 96 --pred_len 96 --d_model 512 --n_heads 8
📁 Project Structure
G_GraphTST/
│
├── 📂 data/                    # Datasets storage
│   ├── solar.csv              # Solar power dataset
│   ├── ETTh1.csv              # Electricity Transformer Temperature dataset
│   └── ...                    # Other benchmark datasets
│
├── 📂 results/                 # Experimental results
│   ├── loss.npy              # Training loss curves (3D array: [samples, pred_len, channels])
│   ├── metrics.npy           # Performance metrics (MAE, MSE, RMSE, MAPE, MSPE, R²)
│   └── ...                    # Model-specific results
│
├── 📂 checkpoints/            # Saved model checkpoints
│   └── model_best.pth        # Best performing model weights
│
├── 📂 test_results/           # Prediction visualization (auto-generated)
│   ├── prediction_curves.pdf # Prediction vs ground truth plots
│   └── ...                    # Additional visualization files
│
├── 📂 models/                 # Model implementations
│   ├── GraphPatchTST.py      # Main G-GraphTST model
│   ├── baselines/            # Baseline model implementations
│   └── ...                    # Other model variants
│
├── 📂 layers/                 # Neural network layer components
│   ├── Transformer_EncDec.py # Transformer encoder-decoder layers
│   ├── SelfAttention_Family.py # Multi-head attention mechanisms
│   ├── Embed.py              # Embedding layers (patch, positional, temporal)
│   └── ...                    # Additional layer implementations
│
├── 📂 utils/                  # Utility functions
│   ├── tools.py              # General utilities (early stopping, learning rate adjustment)
│   ├── metrics.py            # Evaluation metrics calculation
│   ├── timefeatures.py       # Time feature engineering
│   └── ...                    # Other helper functions
│
├── 📂 exp/                    # Experiment workflow
│   ├── exp_basic.py          # Base experiment class
│   ├── exp_long_term_forecasting.py # Long-term forecasting experiments
│   └── ...                    # Other experiment types
│
├── 📄 run.py                  # Main program entry point
├── 📄 requirements.txt       # Python dependencies
├── 📄 README.md              # This file
└── 📄 LICENSE                # License information
🔧 Data Processing Pipeline
1. Data Preprocessing
Our preprocessing pipeline ensures data compatibility and optimal model performance:

Data Cleaning: Remove inconsistencies and handle missing values
Normalization: Apply statistical normalization for stable training
Dataset Splitting: Chronological split (70% train, 10% validation, 20% test)

2. Sliding Window Operation
Time series forecasting relies on sliding window preprocessing:
python# Example sliding window configuration
seq_len = 96      # Input sequence length (96 time steps)
pred_len = 96     # Prediction sequence length (96 time steps) 
stride = 1        # Sliding window stride
For detailed sliding window principles, refer to: Time Series Sliding Window Guide

3. Data Loading
Processed data flows through PyTorch DataLoader for efficient batch processing and model training.
Primary Datasets
Solar Energy Dataset 🌞
Source: NREL Solar Power Data for Integration Studies
Link: http://www.nrel.gov/grid/solar-power-data.html
Description: Solar power production records from 137 PV plants in Alabama State (2006)
Sampling: 10 minutes, 52,560 time steps
ETT (Electricity Transformer Temperature) ⚡
Source: ETT Dataset Repository
Link: https://github.com/zhouhaoyi/ETDataset
Variants: ETTh1, ETTh2 (hourly), ETTm1, ETTm2 (15-minute)
Description: Electricity transformer temperature and load data
PV Live Dataset 🔋
Source: Sheffield Solar PV Live
Link: https://www.solar.sheffield.ac.uk/pvlive/
Description: Real-time PV generation data (July 2024 - July 2025)
Sampling: 30 minutes
Benchmark Datasets
Electricity Load Diagrams 💡
Source: UCI Machine Learning Repository
Link: https://archive.ics.uci.edu/ml/datasets/ElectricityLoadDiagrams20112014
Description: Electricity consumption of 321 clients (2012-2014)
Sampling: 15 minutes → converted to hourly
Traffic Dataset 🚗
Source: CalTrans Performance Measurement System (PeMS)
Link: http://pems.dot.ca.gov
Description: Road occupancy rates from San Francisco Bay area sensors
Variables: 862 sensors, hourly sampling
Exchange Rate Dataset 💱
Source: Multivariate Time Series Data Repository
Link: https://github.com/laiguokun/multivariate-time-series-data
Description: Daily exchange rates of 8 countries (1990-2016)
Countries: Australia, Britain, Canada, Switzerland, China, Japan, New Zealand, Singapore
