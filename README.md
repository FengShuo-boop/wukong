<p align="middle">
  <img src="assets/scaling-law.png" width="300"/>
  <img src="assets/architecture.png" width="300"/>
</p>


# Wukong for large-scale recommendation

Unofficial implementation of the paper [Wukong: Towards a Scaling Law for Large-Scale Recommendation](https://arxiv.org/abs/2403.02545) from Meta.

It presents a novel state-of-the-art architecture for recommendation systems that additionally follows a similar scaling law of large language models, where the model performance seems to increase with respect to the model scale without a clear asymptote on the scales explored in the paper.

This repository contains implementations for both Pytorch and Tensoflow.

## install
```bash
pip install -r requirement.txt
```

## Usage <a name = "usage"></a>

### Dataset prepare

1. download dataset from baidu netdisk: https://pan.baidu.com/s/1fxTInhCjw8uASJd3v79Xog?pwd=xp33
2. change exp/train_*.py code `NPZ_FILE_PATH` to dataset local path

### Pytorch train

```bash
# export to onnx
python3 -m exp.export_torch_to_onnx

# train on criteo kaggle dataset by torch
## single GPU
python3 -m exp.train_torch_wukong_on_criteo_kaggle_dataset
## DDP
torchrun --nproc_per_node=8 -m exp.train_torch_wukong_on_criteo_kaggle_dataset
```

### Tensorflow(v2.6.1) train

- If use musa device, please:
```bash
# clone tensorflow musa extension
git clone git@github.com:MooreThreads/tensorflow_musa_extension.git
cd tensorflow_musa_extension

# build tensorflow musa
bash build.sh

# add absolute path of build/libmusa_plugin.so to exp/ptrain_tensorflow_*.py line 2
# like: `tf.load_library("/path/to/libmusa_plugin.so")`
# and then, use musa device train tensorflow
```
- train

```bash
# export to onnx
python3 -m exp.export_tensorflow_to_onnx

# train on criteo kaggle dataset by tensorflow
python3 -m exp.train_tensorflow_wukong_on_criteo_kaggle_dataset
```

## Citations

```bibtex
@misc{zhang2024wukong,
      title={Wukong: Towards a Scaling Law for Large-Scale Recommendation}, 
      author={Buyun Zhang and Liang Luo and Yuxin Chen and Jade Nie and Xi Liu and Daifeng Guo and Yanli Zhao and Shen Li and Yuchen Hao and Yantao Yao and Guna Lakshminarayanan and Ellie Dingqiao Wen and Jongsoo Park and Maxim Naumov and Wenlin Chen},
      year={2024},
      eprint={2403.02545},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```
