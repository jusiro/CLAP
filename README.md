# CLass adaptive Linear Probing (*CLAP*)
The official implementation of [*A Closer Look at the Few-Shot Adaptation of Large Vision-Language Models*](https://arxiv.org/abs/2312.12730).<br/>
[Julio Silva-Rodriguez](https://scholar.google.es/citations?user=1UMYgHMAAAAJ&hl),
[Sina Hajimiri](https://scholar.google.com/citations?user=C5k-mOYAAAAJ&hl),
[Ismail Ben Ayed](https://scholar.google.es/citations?user=29vyUccAAAAJ&hl),
[Jose Dolz](https://scholar.google.es/citations?user=yHQIFFMAAAAJ&hl)
<br/>
[ÉTS Montreal](https://liviamtl.ca/)
<br/>
| [Project](https://jusiro.github.io/projects/clap) | [Paper](https://arxiv.org/pdf/2312.12730.pdf) | [Code](https://github.com/jusiro/CLAP) |
<br/>

When **adapting CLIP** using only few-shot, it is **unrealistic** to assume the presence of a **validation subset** to empirically
fix a set of hyperparameters per task, *i.e.* model selection. We propose two solutions, which do not require any hyperparameter 
tuning, and thus is adapted strictly using only the support samples.

- A revisited **zero-shot initialized Linear Probe (ZS-LP)**, tailored for CLIP-alike vision-language models.
- A constraint formulation to retain prior knowledge of the robust zero-shot prototypes per class,
  **CLass adaptive Linear Probing (CLAP)**.

## Installation
This repository requires to install the environment and datasets:
- follow [here](https://github.com/KaiyangZhou/Dassl.pytorch#installation) to install [Dassl.pytorch](https://github.com/KaiyangZhou/Dassl.pytorch) and PyTorch.
- run `pip install -r requirements.txt` under `CLAP/` to install a few more packages required by [CLIP](https://github.com/openai/CLIP) (this should be done when `dassl` is activated).
- follow [DATASETS.md](DATASETS.md) to install the datasets.

*PS: You can also follow [CoOp](https://github.com/KaiyangZhou/CoOp) to perform the installation.*

## Usage
We present the basic usage here.

(a) Zero-shot initialized Linear Probe (ZS-LP):
- `bash scripts/adapt.sh 0 imagenet SGD_lr1e-1_B256_ep300 1 ZS none RN50`

(b) CLass adaptive Linear Probing (CLAP):
- `bash scripts/adapt.sh 0 imagenet SGD_lr1e-1_B256_ep300 1 ZS l2 RN50`

(c) Test domain generalization:
- `bash scripts/eval.sh 0 imagenet imagenetv2 SGD_lr1e-1_B256_ep300 1 ZS l2 RN50`

## Acknowledgment
This repository is mainly based on [CoOp](https://github.com/KaiyangZhou/CoOp) and [TaskRes](https://github.com/geekyutao/TaskRes) code base. We sincerely thank prior authors on this topic for his awesome code base.

# Citation

If you find this repository useful, please consider citing this paper:
```
@inproceedings{clap24,
    title={A Closer Look at the Few-Shot Adaptation of Large Vision-Language Models},
    author={Julio Silva-Rodr\'iguez and Sina Hajimiri and Ismail Ben Ayed and Jose Dolz},
    booktitle={IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    pages={23681-23690},
    year={2024}
    }
