# Visual Relation Grounding in Videos

This is the pytorch implementation of our work at ECCV2020 (Spotlight). 
![teaser](https://github.com/doc-doc/vRGV/blob/master/introduction.png)
The repository mainly includes 3 parts: (1) Extract RoI feature; (2) Train and inference; and (3) Generate relation-aware trajectories.
![](https://github.com/doc-doc/vRGV/blob/master/model.png)

## Environment

Anaconda 3, python 3.6.5, pytorch 0.4.1 (Higher version is OK once feature is ready) and cuda >= 9.0. For others libs, please refer to the file requirements.txt.

## Install
Please create an envs for this project using anaconda3 (should install [anaconda](https://docs.anaconda.com/anaconda/install/linux/) first)
```
>conda create -n envname python=3.6.5 # Create
>conda activate envname # Enter
>pip install -r requirements.txt # Install the provided libs
>sh vRGV/lib/make.sh # Set the environment for detection
```
## Data Preparation
Please download the data [here](https://drive.google.com/file/d/1qNJ3jBPPoi0BPkvLqooS66czvCxsib1M/view?usp=sharing). The folder [ground_data] should be at the same directory as vRGV [this project]. Please merge the downloaded vRGV folder with this repo. 

Please download the raw videos [here](https://xdshang.github.io/docs/imagenet-vidvrd.html), and extract them into ground_data/vidvrd/JPEGImages/. 
```
ffmpeg -i vname.mp4 -start_number 0 ./%06d.JPEG
```
The directory should be like: JPEGImages/ILSVRC2015_train_xxx/000000.JPEG.(Please make sure that the index starts from 0.)

## Usage
Feature Extraction (need about 100G storage! Because I dumped all the detected bboxes along with their features. It can be greatly reduced by changing detect_frame.py to return the top-40 bboxes and save them with h5py file.)
```
>./detection.sh
```
Training
```
>./ground.sh 0 train # Train the model with GPU id 0
```
Inference
```
>./ground.sh 0 val # Output the relation-aware spatio-temporal attention
>python generate_track_link.py # Generate relation-aware trajectories with Viterbi algorithm
>python eval_ground.py # Evaluate the performance
```
## Visualization
|Query| bicycle-jump_beneath-person       | person-feed-elephant          | person-stand_above-bicycle       | dog-watch-turtle|
|:---| --------------------------------- | ----------------------------- | ---------------------------------------- | ---------------------------------------- | 
|Result| ![](https://media.giphy.com/media/htciIcJZ2q7pb06zoI/giphy.gif) | ![](https://media.giphy.com/media/dX34r2BJNjVCNCuFNy/giphy.gif)   | ![](https://media.giphy.com/media/ln7xmvrkjcX47W9Kax/giphy.gif)|![](https://media.giphy.com/media/h5uiVR9ukJLVRgT9yC/giphy.gif)|
|Query| person-ride-horse       | person-ride-bicycle          |   person-drive-car     |  bicycle-move_toward-car|
|Result| ![](https://media.giphy.com/media/J5jSa7lJxwFXorWYbx/giphy.gif) | ![](https://media.giphy.com/media/lSsztYWamp6gLfHSfg/giphy.gif)   | ![](https://media.giphy.com/media/S5Kp8KaApxrazkVmcd/giphy.gif)|![](https://media.giphy.com/media/ZE4vFIjfm1BHXP7w0R/giphy.gif)|

## Note  
If you find the code useful in your research, please kindly cite:

```
@inproceedings{xiao2020visual,
  title={Visual Relation Grounding in Videos},
  author={Xiao, Junbin and Shang, Xindi and Yang, Xun and Tang, Sheng and Chua, Tat-Seng},
  booktitle={European Conference on Computer Vision},
  pages={447--464},
  year={2020},
  organization={Springer}
}
```

## License

NUS © [NExT++](https://nextcenter.org/)
