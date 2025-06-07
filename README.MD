# BioFlexNet: Spike-Driven Dynamic Routing for Robust and Biologically-Plausible Computation

Repository for the SpikingFlexNet project under [Kozlov Lab](https://www.kozlovlab.com/), by Zifan Ning.

BioFlexNet is a spiking neural network (SNN) framework that integrates dynamic routing via PlasticFlexLayer, enabling context-dependent path switching between selectivity and invariance.
Built with surrogate gradient training (STBP), BioFlexNet improves optimization stability and enhances adversarial robustness, while maintaining biological plausibility through event-driven, spike-based computation.

The framework supports both ANN and SNN modes, and provides tools for robustness evaluation (FGSM, PGD attacks) and network visualization, targeting visual recognition tasks such as CIFAR-10 and ImageNet100.

![framework](public/framework.jpg)

## Features

- Unified support for ANN, SNN, and STBP-based training
- Customizable convolution-to-pooling ratios and activation masking
- Batch configuration launcher with multiprocessing
- Adversarial attack evaluation (PGD, FGSM, SPSA, etc.)
- Gradient flow, Hessian spectrum, loss landscape, and intermediate activation analysis
- Modular design for fast experimentation and reproducibility

## Dataset Support

FlexNet supports a range of standard image classification datasets. Each dataset can be specified via the `"dataset"` field in the configuration file.

| Dataset      | Num Classes | Images per Class | Num Training Images | Num Validation Images |
|--------------|-------------|------------------|---------------------|-----------------------|
| MNIST        | 10          | 7000             | 60000               | 10000                 |
| CIFAR-10     | 10          | 6000             | 50000               | 10000                 |
| ImageNet100  | 100         | ~1300            | ~130,000            | ~5,000                |
| ImageNet1K   | 1000        | ~1300            | 1,281,167           | 50,000                |

### Notes

- MNIST and CIFAR-10 are loaded directly from torchvision.
- ImageNet100 is a curated subset of 100 classes from ImageNet-1K, provided as a local dataset folder.
- ImageNet1K support assumes the standard ILSVRC2012 directory format (`train/` and `val/` folders with class subdirectories).
- To use custom datasets, you may extend the loader in `src/training/dataset_select.py`.




## Quick Start

1. Clone the repository

```bash 
git clone git@github.com:AaronInKCL/FlexNet.git
```

2. Install dependencies

```bash 
pip install -r requirements.txt
```

3. Run a training example

```bash 
python train_initialiser.py --config configs/example_config.json
```

4. Evaluate models 

```bash 
python src/analysis/run_analysis_suite.py --checkpoint_dir checkpoints/cifar10_Model_SNN_NoSTBP_NoFLEX
```

## Configuration Format

FlexNet uses a JSON-based configuration file to define model architecture, training strategy, and dataset parameters. Below is a minimal example:


```json
{
  "network": "VGG11",
  "dataset": "cifar10",
  "batch_size": 256,
  "learning_rate": 0.01,
  "num_classes": 10,
  "use_flex": true,
  "optimizer": "SGD",
  "logits_mechanism": "THRESHOLD",
  "masking_mechanism": "SIGMOID_HARD",
  "logits_use_batchnorm": true,
  "dropout": 0.2,
  "in_dimensions": [3, 32, 32],
  "conv_pool_ratio": [2, 1],

  "spiking": false,
  "spiking_threshold": 1.0,
  "spiking_decay": 0.95,

  "use_stbp": true
}
```

## Experiments and Analysis

This project includes a set of standardized experiments to evaluate the behavior, robustness, and dynamics of spiking and non-spiking neural architectures. Below are the supported experimental modules and sample visualizations.

> All analysis modules can be triggered in sequence with a single command using the `run_analysis_suite.py` script.

```bash 
python src/analysis/run_analysis_suite.py --checkpoint_dir checkpoints/cifar10_Model_SNN_NoSTBP_NoFLEX
```

### 1. Training Accuracy Comparison

We provide a grid of training runs across different configurations:

- `spiking = [True, False]`
- `use_STBP = [True, False]`
- `mask_type = [soft, hard, none]`
- `conv_pool_ratio = [(1,1), (2,1), (3,1)]`

Training accuracy and test performance can be aggregated from logs stored under `checkpoints/`.

**Example plot:**

![Training Accuracy](public/cifar10-vgg16-combined-accuracy-loss.png)

---

### 2. Adversarial Robustness Evaluation

We evaluate each trained model against common white-box adversarial attacks using `run_attack_collect.py`. Supported attack methods include:

- PGD (Projected Gradient Descent)
- FGSM (Fast Gradient Sign Method)
- SPSA (Simultaneous Perturbation Stochastic Approximation)
- Jitter or randomized attacks (if enabled)

**Example: PGD accuracy curve**

```bash
python run_attack_collect.py \
    --checkpoint_dir checkpoints/cifar10_Model_SNN_STBP_NoFLEX \
    --attack PGD \
    --top_n 1 \
    --sample_size 128
```

![Adversarial Robustness](public/combined_attack_plot_final_customnames.png)

3. Gradient Flow Analysis
We provide a gradient flow diagnostic tool that helps detect vanishing or exploding gradients during backpropagation.


```bash
python src/analysis/gradient_flow.py --checkpoint_dir <dir>
```
**Example plot:**

![Gradient Flow](results/cifar10_Model_SNN_NoSTBP_NoFLEX/gradient_flow.png)

4. Hessian Spectrum and Loss Sharpness

We compute the top eigenvalues of the Hessian to assess the sharpness of the loss landscape.

```bash
python src/analysis/hessian.py --checkpoint_dir <dir>
```
**Example plot:**
![hessian](results/cifar10_Model_SNN_NoSTBP_NoFLEX/hessian/hessian_spectrum_density.png)


5. Intermediate Representation Visualization

We visualize the internal feature maps of the network across layers.

```bash 
python src/analysis/intermediate_plot.py --checkpoint_dir <dir>
```

**Example plot:**
![inte](results/cifar10_Model_SNN_NoSTBP_NoFLEX/intermediate_representations.png)

6. Loss Surface Visualization
Using perturbations along principal directions in parameter space, we generate 2D loss landscapes:

```bash
python src/analysis/loss_surface.py --checkpoint_dir <dir>
```

**Example plot:**
![loss](public/loss_surface_2.png)