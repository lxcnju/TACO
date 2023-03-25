# TACO
The source code of our works on few-shot learning:
* AAAI 2021 paper: Task Cooperation for Semi-Supervised Few-Shot Learning.


# Content
* Personal Homepage
* Basic Introduction
* Running Tips
* Citation

## Personal Homepage
  * [Homepage](https://www.lamda.nju.edu.cn/lixc/)

## Basic Introduction
  * Semi-Supervised Few-Shot Learning (SS-FSL) is proposed in this paper.
  * Inspired by the idea that unlabeled data can be utilized to smooth the model space in traditional semi-supervised learning, we propose TAsk COoperation (TACO) which takes advantage of unsupervised tasks to smooth the meta-model space.

## Environment Dependencies
The code files are written in Python, and the utilized deep learning tool is PyTorch.
  * `python`: 3.7.3
  * `numpy`: 1.21.5
  * `torch`: 1.9.0
  * `torchvision`: 0.10.0
  * `pillow`: 8.3.1

## Datasets
We provide the dataset including:
  * MiniImagenet, which has 60,000 images for few-shot classification. The dataset path should be configured in `paths.py`.

## Running Tips
  * `python train_semi_taco.py`: running SS-FSL, the hyper-paramters should be configured in the code file.
  * `python train_sup_taco.py`: the proposed TACO could also be applied to Supervised Few-shot Learning, the hyper-paramters should be configured in the code file.


## Citation
  * Han-Jia Ye, Xin-Chun Li , De-Chuan Zhan. Task Cooperation for Semi-Supervised Few-Shot Learning. In: Proceedings of the 35th AAAI Conference on Artificial Intelligence (AAAI'21), online conference, 2021.
  * \[[BibTex](https://dblp.org/pid/246/2947.html)\]
