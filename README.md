**Status:** Archive (code is provided as-is, no updates expected)

# 1D Velocity estimation using neural-networks

Code for reproducing results in ["Seismic velocity estimation: a deep recurrent neural-network approach"](tobeannounced)


## Installation

You should clone this repository

    git clone https://github.com/gfabieno/SeisCL.git

#### a) Use Docker (easiest)

We provide a Docker image that contains all necessary python libraries like Tensorflow
and the seismic modeling code SeisCL.

You first need to install the Docker Engine, following the instructions [here](https://docs.docker.com/install/).
To use GPUs, you also need to install the [Nvidia docker](https://github.com/NVIDIA/nvidia-docker).
For the later to work, Nvidia drivers should be installed.
Then, when in the project repository, build the docker image as follows:

    docker build -t seisai:v0

You can then launch any of the python scripts in this repo as follows:

    docker run --gpus all -it\
               -v `pwd`:`pwd` -w `pwd` \
               --user $(id -u):$(id -g) \
               seisai:v0 Case_article.py --logdir=./Case_article

This makes accessible all gpus (`--gpus all`), mounting the current directory to a
to the same path in the docker (second line), running the docker as the current user
(for file permission), and runs the script `Case_article.py`.

#### b) Install all requirements

It is recommended to create a new virtual environment for this project with Python3.
The main python requirements are:
*   [tensorflow](https://www.tensorflow.org). This project was tested with versions 1.8 to 1.15.
The preferred method of installation is through pip, but many options are available.
*  [SeisCL](https://github.com/gfabieno/SeisCL). Follow the instruction in the README of
the SeisCL repository. Preferred compiling options for this project are api=opencl (use
OpenCL, which is faster than CUDA for small models) and nompi=1, because no MPI parallelization is required.
Be sure to install SeisCL's python wrapper.

Once SeisCL is installed, you can install all other python requirements with

    pip install -r requirements.txt


## Reproducing the results

Figures of the article can be reproduced with the script  `reproduce_results.py`.
This script automates the following steps.

#### 1. Training set creation

To create the synthetic training set, do:

    python Case_article.py --training=0
    
Note that to speed up data creation, several GPUs can be used as follows:
    
    export CUDA_VISIBLE_DEVICES=0; python Case_article.py --training=0
    export CUDA_VISIBLE_DEVICES=1; python Case_article.py --training=0

The same strategy can be applied for the subsequent steps.

#### 2. Testing set creation (1D)

To create the synthetic 1D training set, do:

    python Case_article_test1D.py --testing=0

#### 3. Training

Perform the training  with the following command:

    python Case_article.py --training=1 --logdir=Case_article
    
This will train 16 models, for which logs and results will be stored in `Case_article0`,
`Case_article1` and so on.

#### 4. Plot training loss (Figure 2)

To reproduce Figure 2 for trained model 0:

    python plot_loss.py --logdir=Case_article0 --dataset_path=dataset_article/test/dhmin5layer_num_min10

#### 5. Testing in 1D (Figure 3)

Perform the testing on the 1D test set with

    python Case_article_test1D.py --testing=1 --logdir=Case_article*/4_* --niter=1000

This will test on all the trained models contained in `Case_articleX`, for models at iteration
1000 for the 4th step of the training. The predictions for each model are stored in
`/dataset_article/test/Case_articleX`. Figure 3 is reproduced along the test statistics.

#### 6. Testing in 2D (Figure 4)

To create the 2D test set and perform the testing:

    python Case_article_test2D.py --testing=1 --logdir=Case_article*/4_* --niter=1000

This will create the 2D testing set (may take a while) and test on all the trained models.
Figure 4 is produced along the test statistics.

#### 7. Testing on real data (Figures 5 and 6)

To download and preprocess the real data set:

    cd realdata
    python Process_realdata.py

Then the testing is carried with

    python Case_article_testreal.py --plots=2 --logdir=Case_article*/4_* --niter=1000

This produces Figures 5 and 6.




















