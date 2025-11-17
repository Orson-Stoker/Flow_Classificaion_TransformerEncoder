# Network Traffic Classification using Transformer

A deep learning project for network traffic classification using Transformer architecture, processing PCAP files to extract sequential features and perform multi-class classification.

## Project Structure
.
├── config.json # Model and training configuration
├── data_preprocess.py # Data preprocessing script
├── model.py # Transformer model definition
├── packet_dataset.py # PyTorch Dataset class
├── train.py # Training and cross-validation script
└── README.md



## Features

- **Data Preprocessing**: Extract temporal features from PCAP files
- **Transformer Architecture**: Self-attention based encoder blocks
- **Multi-class Classification**: Supports 8 traffic categories
- **Cross-validation**: 5-fold stratified cross-validation
- **Comprehensive Logging**: Training process and performance metrics tracking


