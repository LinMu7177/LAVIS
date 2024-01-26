```sh
conda create -n test python=3.8
conda activate test

conda install pytorch==2.0.1 torchvision==0.15.2  -c pytorch
pip install -e .


# conda install pytorch==1.11.0 torchvision==0.12.0  -c pytorch
# conda install pytorch==1.12.1 torchvision==0.1.1  -c pytorch
# conda install pytorch==2.1.2 torchvision==0.16.2  -c pytorch

cd ..
sudo rm -r ReLA
sudo git clone https://github.com/LinMu7177/ReLA.git
cd ReLA
python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'

cd gres_model/modeling/pixel_decoder/ops/
sh make.sh

cd ../../../../
pip install -r requirements.txt


cd ../LAVIS
pip install transformers==4.31.0
python evaluate.py --cfg-path lavis/projects/blip2/eval/vqav2_zeroshot_instructblip_eval.yaml
```