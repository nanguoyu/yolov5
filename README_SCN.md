
## Setup env

```bash
mamba create -n yolo_scn python=3.11 pytorch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 pytorch-cuda=12.1 -c pytorch -c nvidia
mamba activate yolo_scn
pip install -r requirements.txt
pip install wandb
```
Then please login to wandb.
```bash
wandb login
```

## Train

### SCN

Change `D` and which layers you want to be SCN in `config_scn_yolov5.json` and run the following command.

```bash
CUDA_VISIBLE_DEVICES=7 python train.py  --project SCN --data data/stanford_dogs.yaml --cfg yolov5s.yaml --weights '' --img 320 --epochs 100 --patience 5 --hyp data/hyps/hyp.scratch-low-stanford-dog.yaml --cache ram --optimizer Adam  --workers 12 --batch-size 128  --device 7 --scn
```


### One4one

Train with 120 degree test angle.

```bash
# CUDA_VISIBLE_DEVICES=7 python train.py  --project one4one --data data/stanford_dogs.yaml --cfg yolov5s.yaml --weights '' --img 320 --epochs 100 --patience 10 --hyp data/hyps/hyp.scratch-low-stanford-dog.yaml --cache ram --optimizer Adam  --workers 12 --batch-size 128  --device 7 --test-angle 120
```

### One4all

Train with random agrees from 0 to 360 degree.

```bash
# CUDA_VISIBLE_DEVICES=7 python train.py  --project one4all --data data/stanford_dogs.yaml --cfg yolov5s.yaml --weights '' --img 320 --epochs 100 --patience 10 --hyp data/hyps/hyp.scratch-low-stanford-dog.yaml --cache ram --optimizer Adam  --workers 12 --batch-size 128  --device 7 --test-angle random
```
