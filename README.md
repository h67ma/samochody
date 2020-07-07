# Introduction

This repository contains a modified implementation of ECCV 2018 paper "License Plate Detection and Recognition in Unconstrained Scenarios".

Paper webpage: http://sergiomsilva.com/pubs/alpr-unconstrained/

If you use results produced by our code in any publication, please cite this paper:

```
@INPROCEEDINGS{silva2018a,
  author={S. M. Silva and C. R. Jung}, 
  booktitle={2018 European Conference on Computer Vision (ECCV)}, 
  title={License Plate Detection and Recognition in Unconstrained Scenarios}, 
  year={2018}, 
  pages={580-596}, 
  doi={10.1007/978-3-030-01258-8_36}, 
  month={Sep},}
```

# Modifications made in this fork
**TODO**

# Setup
## Requirements
Verified working environment:
* Ubuntu 18.04
* nvidia-driver-430
* CUDA 10.0 (installed from .deb)
* libcudnn7_7.4.2.24-1 (installed from .deb)
* python 2.7
* tensorflow-gpu 1.14.0
* keras 2.3.1
* opencv-python 4.2.0.32
* numpy 1.16.6

## Build Darknet

```
$ cd darknet && make
```

## Download trained models

```
$ ./get-networks.sh
```

# Training the LP detector

To train the LP detector network from scratch, or fine-tuning it for new samples, you can use the train-detector.py script. In folder samples/train-detector there are 3 annotated samples which are used just for demonstration purposes. To correctly reproduce our experiments, this folder must be filled with all the annotations provided in the training set, and their respective images transferred from the original datasets.

The following command can be used to train the network from scratch considering the data inside the train-detector folder:

```shellscript
$ mkdir models
$ python create-model.py eccv models/eccv-model-scracth
$ python train-detector.py --model models/eccv-model-scracth --name my-trained-model --train-dir samples/train-detector --output-dir models/my-trained-model/ -op Adam -lr .001 -its 300000 -bs 64
```

For fine-tunning, use your model with --model option.
