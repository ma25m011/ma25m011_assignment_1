# DA6401 Assignment 1 – Multi-Layer Perceptron for Image Classification

## Links
- **W&B Report:** https://wandb.ai/ma25m011-ii/da6401-assignment-1/reports/DA6401-Assignment-1--VmlldzoxNjEzNDY3NQ?accessToken=7qpn7rbq0l0qpop6p2oheaxoc9jp3q6qdw5escnrutis7ti0xawr1lnvnrxprp6y 
- **GitHub Repository:** https://github.com/ma25m011/da6401_assignment_1

## Project Structure
```
├── README.md
├── requirements.txt
├── models/
├── notebooks/
│   └── wandb_analysis.py
└── src/
    ├── best_model.npy
    ├── best_config.json
    ├── train.py
    ├── test.py
    ├── inference.py
    ├── ann/
    │   ├── activations.py
    │   ├── neural_layer.py
    │   ├── neural_network.py
    │   ├── objective_functions.py
    │   └── optimizers.py
    └── utils/
        ├── data_loader.py
        └── metrics.py
```

## Setup
```bash
pip install -r requirements.txt
```

## Training
```bash
python src/train.py \
  --dataset fashion_mnist \
  --epochs 20 \
  --optimizer rmsprop \
  --learning_rate 0.001 \
  --num_layers 3 \
  --hidden_size 128 128 64 \
  --activation relu \
  --weight_init xavier \
  --wandb_project da6401-assignment-1
```

## Inference
```bash
python src/inference.py --dataset fashion_mnist
```

## W&B Analysis (all report sections)
```bash
python notebooks/wandb_analysis.py --wandb_project da6401-assignment-1
```

## Hyperparameter Sweep
```bash
python wandb/sweep.py --wandb_project da6401-assignment-1 --count 100
```
