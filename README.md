# PredRNN++
This is a TensorFlow implementation of [PredRNN++](https://arxiv.org/abs/1804.06300), a recurrent model for video prediction as described in the following paper:

**PredRNN++: Towards A Resolution of the Deep-in-Time Dilemma in Spatiotemporal Predictive Learning**, by Yunbo Wang, Zhifeng Gao, Mingsheng Long, Jianmin Wang and Philip S. Yu.

## Setup
Required python libraries: tensorflow (>=1.0) + opencv + numpy.\
Tested in ubuntu/centOS + nvidia titan X (Pascal) with cuda (>=8.0) and cudnn (>=5.0).

## Datasets
We conduct experiments on three video datasets: [Moving Mnist](https://1drv.ms/f/s!AuK5cwCfU3__fGzXjcOlzTQw158), [Human3.6M](http://vision.imar.ro/human3.6m/description.php), [KTH Actions](http://www.nada.kth.se/cvap/actions/).\
For video format datasets, please extract frames from original video clips and move them to the `data/` folder.

## Training
Use the train.py script to train the model. To train the default model on Moving MNIST simply use:
```
python train.py
```
You might want to change the `--train_data_paths`, `--valid_data_paths` and `--save_dir` which point to paths on your system to download the data to, and where to save the checkpoints.

To train on your own dataset, have a look at the `InputHandle` classes in the `data_provider/` folder. You have to write an analogous iterator object for your own dataset. 

At inference, the generated future frames will be saved in the `--results` folder.

## Prediction samples
The ground truth | PredRNN++ | A baseline model.\
10 frames are predicted given the last 10 frames.

<div align=center><img width="192" height="64" src="https://github.com/Yunbo426/ImageToGit/blob/master/23.gif"/></div>

## Citation
Please cite the following paper if you find this repository useful.
```
@inproceedings{wang2018predrnn,
    title={PredRNN++: Towards A Resolution of the Deep-in-Time Dilemma in Spatiotemporal Predictive Learning},
    author={Wang, Yunbo and Gao, zhifeng and Long, Mingsheng and Wang, Jianmin and Yu, Philip S.},
    journal={ICML},
    year={2018}
}
```

