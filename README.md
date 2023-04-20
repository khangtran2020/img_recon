# Image Reconstruction attacks on ScalableRR


## Requirement

```angular2html
tensorflow >= 2.4.x
numpy
matplotlib
PIL
```

## Data
+ Run `bash init.sh`
+ The CelebA dataset can be downloaded from here https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html
+ Download the `img_align_celeba` folder from the link and put the folder into the `Data` directory.

## Run the attack:

+ For the attack from [1], please run the experiments in the file `Inference_CelebA.ipynb`. Please note that, this source code already contains the trained model based on LFW dataset [3] in directory `model_lfw`.
+ For the attack from [2], please run the experiments in the file `generative_model.ipynb`.

## References:

[1] T. Brox A. Dosovitskiy. 2016. Inverting Visual Representations with Convolutional Networks.
CVPR (2016)

[2] H. Oh and Y. Lee. 2019. Exploring image reconstruction attack in deep learning computation
offloading. In EMDL.

[3] Gary B. Huang, Manu Ramesh, Tamara Berg, and Erik Learned-Miller. Labeled Faces in the Wild: A Database for Studying Face Recognition in Unconstrained Environments. University of Massachusetts, Amherst, Technical Report 07-49, October, 2007.