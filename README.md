# prox-svrg
Implementation of the Proximal Stochastic Variance Reduced Gradients Algorithm.

# Dataset Details
-------------------
- Download the dataset required for the analysis from the following locations:
  - [rcv1](https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/rcv1_train.binary.bz2)
  - [covertype](https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/covtype.libsvm.binary.scale.bz2)
  - [sido0](http://www.causality.inf.ethz.ch/data/sido0_text.zip)

# Setup instructions
----------------------
- From the root of the directory, execute the following:
  ```bash
  bash ./utils/download_dataset.sh
  ```

# Test scripts
---------------
- test_dataset_loader.py: Processes the dataset.
- test_loss.py: Tests if the loss functions return appropriate values.
- test_optimizers.py: Tests if the optimizers work.
