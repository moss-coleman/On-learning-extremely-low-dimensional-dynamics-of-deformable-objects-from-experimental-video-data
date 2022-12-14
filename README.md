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

To install the dependencies, it is recommended to used the `Project.toml` file to create a project environment.

To setup the environment, `git clone` the package and `cd` to the project directory. Then call :

``` bash
(v1.8) pkg> activate .

(On-learning-extremely-low-dimensional-dynamics-of-deformable-objects-from-experimental-video-data) pkg> instantiate
```

## Training the VAE model

To train a VAE model, select the data set you want to train on and the training parameters from the `Args` struct in the `src/train_VAE.jl` file. 

``` Julia

@with_kw mutable struct Args
    η = 1e-4                # learning rate
    λ = 0.01f0              # regularization paramater
    epochs = 5              # number of epochs
    seed = 42               # random seed
    cuda = true             # use GPU
    input_dim = [36, 64, 1] # image size
    latent_dim = 1          # latent dimension
    beta = 1.0              # β value in loss function
    variation = "shape"     # what to vary, "shape" or "length"
    data_set = "concave"    # what set of states to train the VAE to represent
    filter_width = 4        # CNN kernal dimension
end

```

To train the model, in julia REPL:

```
julia> include("train_VAE.jl")
```

## Training the dynamic model

Firstly, you need to use the encoder from the VAE model trained on the same data set to encode the data to the latent space, in format that has inputs to the model and corresponding output data labelled. To do this, set the appropriate parameters, the `Args_data` struct, in the `/src/data_pre_processing.jl` file. Before training the dynamic model, run the file:

```
julia> include("data_pre_processing.jl")
```

For the dynamic model in the latent space, select the training parameters from the `Args_mlp` struct in the `train_MLP.jl` file.


```
julia> include("train_MLP.jl")
```
