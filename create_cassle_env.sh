 #!/bin/bash 

eval "$(conda shell.bash hook)"

conda create --name cassle python=3.8

conda init

conda activate cassle

conda install pytorch=1.10.2 torchvision cudatoolkit=11.3 -c pytorch

pip install pytorch-lightning==1.5.4 lightning-bolts wandb sklearn einops

pip install --extra-index-url https://developer.download.nvidia.com/compute/redist --upgrade nvidia-dali-cuda110
