## Usage

- ### Installation

Python 3.8 + PyTorch 1.9.0 + CUDA 11.1 + Detectron2 (v0.6)
```
git clone https://github.com/ViTAE-Transformer/DeepSolo.git
cd DeepSolo
conda create -n deepsolo python=3.8 -y
conda activate deepsolo
pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html
pip install -r requirements.txt
python -m pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu111/torch1.9/index.html
python setup.py build develop
```


- ### Datasets
```
|- DeepSolo 
   |-datasets
        |- maps
        |  |- train
        |  |- val
        |  └  map-anno
        |       |- train
        |       └  val
        |- totaltext
        |  |- train_images
        |  |- test_images
        |  |- train_37voc.json
        |  |- train_96voc.json
        |  └  test.json
```

- ### Pretrained model weights

ViTAEv2-S:

https://onedrive.live.com/?authkey=%21AMIuN9rorTIEzPs&id=E534267B85818129%2125597&cid=E534267B85818129&parId=root&parQt=sharedby&o=OneUp

R-50:

https://onedrive.live.com/?authkey=%21AFGOhAtfJCL29hw&id=E534267B85818129%2125585&cid=E534267B85818129&parId=root&parQt=sharedby&o=OneUp

