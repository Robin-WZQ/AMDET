# 🛡️ Why Assimilation Matters: Rethinking Textual Backdoor Defense in Vision-Language Pretrained Models

> [Zhongqi Wang](https://scholar.google.com.hk/citations?hl=zh-CN&user=Gi1brbgAAAAJ), [Jie Zhang*](https://scholar.google.com.hk/citations?user=hJAhF0sAAAAJ&hl=zh-CN), [Shiguang Shan](https://scholar.google.com.hk/citations?hl=zh-CN&user=Vkzd7MIAAAAJ), [Xilin Chen](https://scholar.google.com.hk/citations?hl=zh-CN&user=vVx2v20AAAAJ)
>
> *Corresponding Author

## 🔥 News

- [2025/9/5] We release all the source code for reproducing the results in our paper.

## 👀 Overview



## 🧭 Getting Start

### Environment Requirement 🌍

T2Ishield has been implemented and tested on Pytorch 2.2.0 with python 3.10. It runs well on both Windows and Linux.

1. Clone the repo:

   ```
   git clone https://github.com/Robin-WZQ/T2IShield
   cd T2IShield
   ```

2. We recommend you first use `conda` to create virtual environment, and install `pytorch` following [official instructions](https://pytorch.org/).

   ```
   conda create -n T2IShield python=3.10
   conda activate T2IShield
   python -m pip install --upgrade pip
   pip install torch==2.2.0+cu118 torchvision==0.17.0+cu118 --extra-index-url https://download.pytorch.org/whl/cu118
   ```

3. Then you can install required packages thourgh:

   ```
   pip install -r requirements.txt
   ```

## 🏃🏼 Running Scripts

### Backdoor Detection🔎

1. Generate the data:

```
cd Utils
python preprocess.py
```

The data structure here should be like:

```
|-- Data
    |-- Main
    	|-- Features
    		|-- CausalAttention
    		|-- FirstLayer
    		|-- OriginalFeature
    	|-- Prompts
    		|-- Prompts.txt
```

2. Detect the model if is backdoored or not:

```
cd ../
python main.py
```

The results file structure should be like:

```
|-- Results
    |-- Model_name
    	|-- Images # 4 images that contain the backdoor target semantic
    	|-- Backdoor_Embedding.pt # optimized embedding in the first layer
		|-- Backdoor_Feature.pt # last layer feature
		|-- log.txt
```

### Backdoor Mitigation⚒️

> Here, we 

please follow the instruction in xxx.

## 📄 Citation

If you find this project useful in your research, please consider cite:

```
@article{wang2025xxx,
title={xxx}, 
author={Zhongqi Wang and Jie Zhang and Shiguang Shan and Xilin Chen},
journal={xxx},
year={2025},
}
```

🤝 Feel free to discuss with us privately!
