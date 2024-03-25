## Usage

- ### Installation

Python 3.8 + PyTorch 1.9.0 + CUDA 11.1 + Detectron2 (v0.6)
```
git clone https://github.com/prajolshrestha/DeepSolo.git
cd DeepSolo
conda create -n deepsolo python=3.8 -y
conda activate deepsolo
pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html
pip install -r requirements.txt
python -m pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu111/torch1.9/index.html
python setup.py build develop
```


- ### Datasets
Download dataset:


```
|- DeepSolo 
   |-datasets
        |- maps
        |  |- train
        |  |- val
        |  └  map-anno
        |       |- train
        |       └  val
```

- ### Pretrained model weights

#### ViTAEv2-S:
Download and insert inside following path:
"output/ViTAEv2_S/TotalText/pretrain/vitaev2-s_pretrain_synth-tt-mlt-13-15-textocr.pth"

https://onedrive.live.com/?authkey=%21AMIuN9rorTIEzPs&id=E534267B85818129%2125597&cid=E534267B85818129&parId=root&parQt=sharedby&o=OneUp

Finetune command: 
$ python tools/train_net.py --config-file configs/ViTAEv2_S/Map/finetune_map.yaml --num-gpus 1



#### R-50:
Download and insert inside following path:
"output/R50/150k_tt/pretrain/res50_pretrain_synth-tt.pth"

https://onedrive.live.com/?authkey=%21AFGOhAtfJCL29hw&id=E534267B85818129%2125585&cid=E534267B85818129&parId=root&parQt=sharedby&o=OneUp

Finetune command:
$ python tools/train_net.py --config-file configs/R_50/Map/finetune_map.yaml --num-gpus 1
