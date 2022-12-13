# On-learning-extremely-low-dimensional-dynamics-of-deformable-objects-from-experimental-video-data

This repository is for the code accompaning the paper "On learning extremely low dimensional dynamics of deformable objects from experimental video data", submitted to Learning for Dynamics and control 2023. 

It contains the experimental data used in the paper, the code for processing, the script used to train the VAE models and script for training the MLP in the latent space. 

## Overview of overall architecture 
<!-- ![Alt text](./overall_architecture.svg) -->
<img src="./overall_architecture.svg" width="700" />


## Dependencies 

Operating System - Ubuntu 20.04
Language versions - Julia 1.8.2 (What we used, might work with other versions)

### Julia Dependencies

To install the dependencies, it is recommended to used the Project.toml file to create a project environment.

To setup the environment, `git clone` the package and cd to the project directory. Then call :

``` bash
(v1.8) pkg> activate .

(On-learning-extremely-low-dimensional-dynamics-of-deformable-objects-from-experimental-video-data) pkg> instantiate
```

## Training the VAE model

To train a VAE model, select the data set you want to train on and the training parameters from the Args in the `src/train_VAE.jl` file. 

``` Julia


```


