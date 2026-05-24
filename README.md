# ASSR-Net: Anisotropic Structure-Aware and Spectrally Recalibrated Network for Hyperspectral Image Fusion(ASSR-Net)
![Language](https://img.shields.io/badge/language-python-brightgreen) 

## Dataset
* [Cave](https://cave.cs.columbia.edu/repository/Multispectral)
* [Harvard](http://vision.seas.harvard.edu/hyperspec/)

Please place the downloaded datasets in the following directory structure:
```text
Dataset/
├── Cave/
│   ├── Train/      # CAVE training images
│   └── Test/       # CAVE testing images
└── Harvard/
    ├── Train/      # Harvard training images
    └── Test/       # Harvard testing images
```
> Note: Each subfolder should contain hyperspectral images in `.mat`

## Environment Setup and Training
```bash
# Create environment
conda create -n assr python=3.11.2
conda activate assr

# Install dependencies
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
pip install numpy scipy matplotlib h5py scikit-image pandas tqdm thop hdf5storage

# Train on CAVE dataset
python train_cave.py

# Train on Harvard dataset
python train_harvard.py

# Test
python test_cave.py
```
