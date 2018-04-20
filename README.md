# PredRNN++
This is a TensorFlow implementation of [PredRNN++](https://arxiv.org/abs/1804.06300), a recurrent model for video prediction as described in the following paper:

**PredRNN++: Towards A Resolution of the Deep-in-Time Dilemma in Spatiotemporal Predictive Learning**, by Yunbo Wang, Zhifeng Gao, Mingsheng Long, Jianmin Wang and Philip S. Yu.

## Setup
Required python libraries: tensorflow (>=1.0) + opencv + numpy.\
Tested in ubuntu/centOS + nvidia titan X (Pascal) with cuda (>=8.0) and cudnn (>=5.0).

## Datasets
We experimented on three video prediction datasets: [Moving Mnist](https://1drv.ms/f/s!AuK5cwCfU3__fGzXjcOlzTQw158), [Human3.6M](http://vision.imar.ro/human3.6m/description.php), [KTH Actions](http://www.nada.kth.se/cvap/actions/). After download, please move them to the `data/` folder. For video format datasets, please extract frames from original video clips.

## Training
Use the train.py script to train the model. To train the default model on Moving MNIST simply use:
```
python train_network.py
```
You might want to change the `--train_data_paths`, `--valid_data_paths` and `--save_dir` which point to paths on your system to download the data to, and where to save the checkpoints.

To train on your own dataset, have a look at the `InputHandle` classes in the `data_provider/` folder. You have to write an analogous iterator object for your own dataset. 

At inference, the generated future frames will be saved in the `--results` folder.

## Citation
Please cite the following paper if you feel this repository useful.
```
@article{wang2018predrnn,
    title={PredRNN++: Towards A Resolution of the Deep-in-Time Dilemma in Spatiotemporal Predictive Learning},
    author={Wang, Yunbo and Gao, zhifeng and Long, Mingsheng and Wang, Jianmin and Yu, Philip S.},
    journal={arXiv preprint arXiv:1804.06300},
    year={2018}
}
```

