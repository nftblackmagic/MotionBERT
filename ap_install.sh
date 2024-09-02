# 1. Create a conda virtual environment.
conda create -n alphapose python=3.7 -y
conda activate alphapose

# 2. Install PyTorch
conda install pytorch torchvision torchaudio pytorch-cuda=11.3 -c pytorch -c nvidia 

# 3. Get AlphaPose
git clone https://github.com/MVIG-SJTU/AlphaPose.git
cd AlphaPose


# 4. install
export PATH=/usr/local/cuda/bin/:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64/:$LD_LIBRARY_PATH
python -m pip install cython
sudo apt-get install libyaml-dev
################Only For Ubuntu 18.04#################
locale-gen C.UTF-8
# if locale-gen not found
sudo apt-get install locales
export LANG=C.UTF-8
######################################################
python setup.py build develop

# 5. Install PyTorch3D (Optional, only for visualization)
conda install -c fvcore -c iopath -c conda-forge fvcore iopath
conda install -c bottler nvidiacub
pip install git+ssh://git@github.com/facebookresearch/pytorch3d.git@stable
pip install gdown

# Create directories for YOLO and tracker data
mkdir -p detector/yolo/data
mkdir -p detector/tracker/data
mkdir -p pretrained_models

# Download YOLO weights
gdown --id 1D47msNOOiJKvPOXlnpyzdKA3k6E97NTC -O ./detector/yolo/data/yolov3-spp.weights

# Download tracker data
gdown --id 1nlnuYfGNuHWZztQHXwVZSL_FvfE551pA -O ./detector/tracker/data/JDE-1088x608-uncertainty

# Download pre-trained model fast_res50_256x192.pth
gdown --id 1kQhnMRURFiy7NsdS8EFL-8vtqEXOgECn -O ./pretrained_models/fast_res50_256x192.pth

# Download pre-trained model halpe26_fast_res50_256x192.pth
gdown --id 1S-ROA28de-1zvLv-hVfPFJ5tFBYOSITb -O ./pretrained_models/halpe26_fast_res50_256x192.pth

wget -P ./detector/yolox/data/ https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.0/yolox_x.pth
