# 🛡️Assimilation Matters: Model-level Backdoor Detection in Vision-Language Pretrained Models

> [Zhongqi Wang](https://scholar.google.com.hk/citations?hl=zh-CN&user=Gi1brbgAAAAJ), [Jie Zhang*](https://scholar.google.com.hk/citations?user=hJAhF0sAAAAJ&hl=zh-CN), [Shiguang Shan](https://scholar.google.com.hk/citations?hl=zh-CN&user=Vkzd7MIAAAAJ), [Xilin Chen](https://scholar.google.com.hk/citations?hl=zh-CN&user=vVx2v20AAAAJ)
>
> *Corresponding Author

We propose **AMBER**, a model-level textual backdoor defense on pretrained encoders. 

The defender ***DO NOT*** have the knowledge of:

1. the trigger and corresponding target.
2. downstream tasks.
3. pre-training dataset.

Our defense involves both detection and mitigation, requiring only less than ***7 min*** to scan the pretrained encoders.

## 🔥 News

- [2025/9/5] We release all the source code for reproducing the results in our paper.

## 👀 Overview
<div align=center>
<img src='https://github.com/Robin-WZQ/AMBER/blob/main/Images/Background.png' width=500>
</div>


## 🧭 Getting Start

### Environment Requirement 🌍

FAD has been implemented and tested on Pytorch 2.2.0 with python 3.10. It runs well on both Windows and Linux.

1. Clone the repo:

   ```
   git clone https://github.com/Robin-WZQ/FAD
   cd FAD
   ```

2. We recommend you first use `conda` to create virtual environment, and install `pytorch` following [official instructions](https://pytorch.org/).

   ```
   conda create -n FAD python=3.10
   conda activate FAD
   python -m pip install --upgrade pip
   pip install torch==2.2.0+cu118 torchvision==0.17.0+cu118 --extra-index-url https://download.pytorch.org/whl/cu118
   ```

3. Then you can install required packages thourgh:

   ```
   pip install -r requirements.txt
   ```

## 🏃🏼 Running Scripts

### Backdoor Detection🔎

> Scan the model to judge if it is backdoored or not. 
>
> If it is backdoored, return the pseudo-trigger embedding and its target.

- Scan the model

```
python main.py 
```

The results file structure should be like:

```
|-- Results
    |-- Model_name
    	|-- Images # 4 images that contain the backdoor target semantic
    	|-- Backdoor_Embedding_Inversion.pt # optimized embedding which can be loaded by Textual Inversion 
    	|-- Backdoor_Embedding.pt # optimized embedding
        |-- Backdoor_Feature.pt # last layer feature
        |-- log.txt
```

The data structure here would be like:

```
|-- Data
    |-- Main
    	|-- Features
    		|-- HiddenStates
    		|-- OriginalFeature
    	|-- Prompts
    		|-- Prompts.txt
```

- Generate the reversed target images.

```
cd ./Utils
python generate_image_input_features.py
```


### Visualization 🖼️

> We also provide the visualization script for reproducing the images in our paper. 

- Please refer to ```./Analysis``` and follow the specific instruction in each file.
  - coverage.ipynb
  - assimilation.ipynb
  - assimilation_each_layer.ipynb


------

### Backdoor Attack 🦠

> Here, we focus on two scenarios:

- Text-on-Text

  ```
  bash backdoor_injection_text_on_text.sh
  ```

- Image-on-Text

    ```
    bash backdoor_injection_image_on_text.sh
    ```

It will generate backdoored models with specific target. 

To change the hyper-parameters of attacking, please refer to ```./Backdoor_Attack/configs```

### Benign Fine-tuning 😁

> We fine-tune the text encoder on clean dataset, i.e., coco30K.

```
python ./Utils/finetuning_on_coco30k.py
```

## Results

> Here, we provide some results to show the effectiveness of our defense

- Backdoor Detection

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
