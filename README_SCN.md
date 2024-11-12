
## Setup env

```bash
mamba create -n yolo_scn python=3.11 pytorch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 pytorch-cuda=12.1 -c pytorch -c nvidia
mamba activate yolo_scn
pip install -r requirements.txt
pip install wandb
```
