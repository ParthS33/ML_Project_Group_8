Requirements:
* pytorch: 2.2.2
* python: 3.10
* tensorflow:
* numpy: 1.26.4
* nltk: 3.8.1
* skikit-learn: 1.4.2
* tqdm: 4.66.2

GPU is required. You can download the CUDA toolkit from [here]([url](https://developer.nvidia.com/cuda-toolkit)). We used NVIDIA CUDA 12.1

Step 1:
Download BERT-Base (Google's pre-trained models) and then convert a tensorflow checkpoint to a pytorch model.

```
python convert_tf_checkpoint_to_pytorch.py \
--tf_checkpoint_path uncased_L-12_H-768_A-12/bert_model.ckpt \
--bert_config_file uncased_L-12_H-768_A-12/bert_config.json \
--pytorch_dump_path uncased_L-12_H-768_A-12/pytorch_model.bin
```
Step 2:
