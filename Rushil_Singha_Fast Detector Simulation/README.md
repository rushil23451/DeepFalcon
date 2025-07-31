# Graph Representation Learning for Fast Detector Simulation

## Overview
This repository contains the code for my Google Summer of Code 2025 Phase 1 project with Machine Learning for Science (ML4Sci). The project focuses on accelerating particle detector simulations using graph neural networks and latent diffusion models. It extends the DeepFalcon framework to handle multi-layer detectors (Tracker, ECAL, HCAL) by leveraging ChebNet for spatial encoding and a custom Latent Diffusion Model (LDM) for efficient jet generation, achieving up to 400x speedup over traditional Monte Carlo methods while maintaining high fidelity.

## Features
- **ChebNet Encoder**: Implements a four-layer ChebNet with spectral graph convolutions to encode heavy-atom coordinates and k-NN graphs into latent embeddings.
- **Latent Diffusion Model**: Trains on pooled latent embeddings to predict and reverse noise, enhancing simulation efficiency.
- **Enhanced Decoder**: Reconstructs 125×1000 jet grids with improved accuracy using a multi-layer perceptron.
- **Pipeline**: Integrates data preprocessing, graph creation, and batched training for scalable performance.

## Requirements
- Python 3.8+
- PyTorch 2.0.0+
- Torch Geometric (with torch-scatter, torch-sparse, torch-cluster, torch-spline-conv)
- NetworkX
- Scikit-learn
- PyArrow
- Pandas
- Matplotlib

## Install dependencies via:

pip install torch torch-geometric torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-2.0.0+cpu.html

pip install networkx scikit-learn pyarrow pandas matplotlib
