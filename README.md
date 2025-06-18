# MVEB: Self-Supervised Learning With Multi-View Entropy Bottleneck

## Overview
This project implements self-supervised learning with multi-view entropy bottleneck. The repository contains:

- Multi-view data loading pipelines
- Entropy bottleneck model implementations
- Pretraining & linear evaluation scripts
- Entropy estimator is suitable for the characteristics of hypersphere, not suitable for Euclidean space

## Project Structure

```bash
project_root/
├── configs/                # Training configurations
│   └── mveb.py             # MVEB main config
├── project/                # Core implementation
│   ├── __init__.py         # Package initialization
│   ├── dataloader.py       # Multi-view data loading
│   ├── entropy.py          # Entropy computations
│   ├── model.py            # Model architectures
│   ├── pretrainer.py       # Pretraining logic
│   ├── utils.py            # Helper functions
│── lineval.py              # Linear evaluation
│── pretrain.py             # Pretraining entrypoint
│── run_linear.sh           # Linear evaluation script
│── run_pretrain.sh         # Pretraining script
└── logs/                   # Training logs (auto-created)
```

## Usage

### Running Pretraining
To run the pretraining procedure, execute the following shell script:

```bash
./run_pretrain.sh 

#Pre-training on 32 V100s 
```
Or you could run the code manually:
```bash
python run_pretrain.py
```
### Running Linear Evaluation
To run the linear evaluation procedure, execute the following shell script:
```bash
./run_linear.sh
```
Or you could run the code manually:
```bash
python run_linear.py
```

## Requirements
- python >=3.9

## Citation
Please cite our paper if you use the code:
```bash
@ARTICLE{10477543,
  author={Wen, Liangjian and Wang, Xiasi and Liu, Jianzhuang and Xu, Zenglin},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence}, 
  title={MVEB: Self-Supervised Learning With Multi-View Entropy Bottleneck}, 
  year={2024},
  volume={46},
  number={9},
  pages={6097-6108},
  keywords={Task analysis;Entropy;Self-supervised learning;Mutual information;Supervised learning;Representation learning;Feature extraction;Minimal sufficient representation;representation learning;self-supervised learning},
  doi={10.1109/TPAMI.2024.3380065}}

```
