# NExT-QA <img src="images/logo.png" height="64" width="128">

We reproduce some SOTA VideoQA methods to provide benchmark results for our NExT-QA dataset published on CVPR2021 (with 1 Strong Accept and 2 Weak Accepts). 

NExT-QA is a VideoQA benchmark to advance video understanding from describing to explaining the temporal actions. It is unique in that it goes beyond descriptive QA (what is) to benchmark causal and temporal action reasoning (why/how did) in realistice videos and it is also rich in object interactions. NExT-QA contains 5440 videos and over 52K manually annotated question-answer pairs grouped into causal, temporal and descriptive questions. We set up both multi-choice and open-ended QA tasks on the dataset. This repo. provides resources for multi-choice QA. Open-ended QA is found in [NExT-OE](https://github.com/doc-doc/NExT-OE). For more details, please refer to our [dataset](https://github.com/doc-doc/NExT-QA) page.

## Environment

Anaconda 3, python 3.6.8, pytorch 1.6 and cuda 10.2. For others libs, please refer to the file requirements.txt.

## Install
Please create an env for this project using anaconda3 (should install [anaconda](https://docs.anaconda.com/anaconda/install/linux/) first)
```
>conda create -n videoqa python=3.6.8
>conda activate videoqa
>git clone https://github.com/doc-doc/NExT-QA.git
>pip install -r requirements.txt
```
## Data Preparation
Please download the pre-computed features and QA annotations from [here](https://drive.google.com/drive/folders/1gKRR2es8-gRTyP25CvrrVtV6aN5UxttF?usp=sharing). There are three zip files correspond to 1) appearance and motion feature for video representation, 2) finetuned BERT feature for QA-pair representation, and 3) annotations of QAs and GloVe Embeddings. After downloading the data, please create a folder ['data'] at the same directory as NExT-QA (this repo), then unzip the video and QA features into it. You will have directories like ['data/vid_feat/' and 'data/qas_bert/']. Please unzip the file 'nextqa.zip' into ['NExT-QA/dataset/nextqa']. 

You are also encouraged to design your own pre-computed video features. In that case, please download the raw videos from [VidOR](https://xdshang.github.io/docs/vidor.html). As NExT-QA's videos are sourced from VidOR, you can easily link the QA annotations with the corresponding videos according to the key 'video' in the [nextqa/*.csv] files, during which you may need the map file ['nextqa/map_vid_vidorID.json'].


## Usage
Once the data is ready, you can easily run the code. First, to test the environment and code, we provide the prediction and model of the SOTA approach (i.e., HGA) on NExT-QA. 
You can get the results reported in the paper by running: 
```
>python mul_eval.py
```
The command above will load the prediction file under ['results/'] and evaluate it. 
You can also obtain the prediction by running: 
```
>./main.sh 0 val #Test the model with GPU id 0
```
The command above will load the model under ['models/'] and generate the prediction file.
If you want to train the model, please run
```
>./main.sh 0 train # Train the model with GPU id 0
```
It will train the model and save to ['models']
## Accuracy on validation set
| Methods                  | Text Rep. | Acc_C | Acc_T | Acc_D | Acc | Text Rep. | Acc_C | Acc_T | Acc_D | Acc   |
| -------------------------| --------: | ----: | ----: | ----: | ---:| --------: | ----: | ----: | ----: | ----: |
| BlindQA                  |   GloVe   | 26.89 | 30.83 | 32.60 | 30.60 | BERT-FT | 42.62 | 45.53 | 43.89 | 43.76 |
| EVQA                     |   GloVe   | 28.69 | 31.27 | 41.44 | 31.51 | BERT-FT | 42.64 | 46.34 | 45.82 | 44.24 |
| [STVQA](https://openaccess.thecvf.com/content_cvpr_2017/papers/Jang_TGIF-QA_Toward_Spatio-Temporal_CVPR_2017_paper.pdf) [CVPR17]  |   GloVe   | 36.25 | 36.29 | 55.21 | 39.21 | BERT-FT | 44.76 | 49.26 | 55.86 | 47.94 |
| [CoMem](https://openaccess.thecvf.com/content_cvpr_2018/CameraReady/1924.pdf) [CVPR18]  |   GloVe   | 35.10 | 37.28 | 50.45 | 38.19 | BERT-FT | 45.22 | 49.07 | 55.34 | 48.04 |
| [HME](https://openaccess.thecvf.com/content_CVPR_2019/papers/Fan_Heterogeneous_Memory_Enhanced_Multimodal_Attention_Model_for_Video_Question_Answering_CVPR_2019_paper.pdf) [CVPR19]    |   GloVe   | 37.97 | 36.91 | 51.87 | 39.79 | BERT-FT | 46.18 | 48.20 | 58.30 | 48.72 |
| [HGA](https://openaccess.thecvf.com/content_cvpr_2018/CameraReady/1924.pdf) [AAAI20]    |   GloVe   | 35.71 | 38.40 | **55.60** | 39.67 | BERT-FT | **46.26** | **50.74** | **59.33** | **49.74** |
| [HCRN](https://openaccess.thecvf.com/content_CVPR_2020/papers/Le_Hierarchical_Conditional_Relation_Networks_for_Video_Question_Answering_CVPR_2020_paper.pdf) [CVPR20]   |   GloVe   | **39.09** | **40.01** | 49.16 | **40.95** | BERT-FT | 45.91 | 49.26 | 53.67 | 48.20 |
| **Human**                |    -      | **87.61** | **88.56** | **90.40** | **88.38** |  -  | **87.61** | **88.56** | **90.40** | **88.38** |
## Multi-choice QA v.s Open-ended QA
![vis mc_oe](./images/res-mc-oe.png)
## Citation
```
@inproceedings{xiao2021nextqa,
  title={NExT-QA: Next Phase of Question-Answering to Explaining Temporal Actions},
  author={Xiao, Junbin and Shang, Xindi and Angela Yao and Chua, Tat-Seng},
  booktitle={Computer Vision and Pattern Recognition (CVPR)},
  year={2021},
  organization={IEEE}
}
```
## Acknowledgement
Our reproduction of the methods are based on the respective official code repositories, we thanks the authors to release their code. If you use the related parts, please cite the corresponding paper which we have commented in the code.
