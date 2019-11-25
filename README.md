# prox-svrg
Implementation of the [Proximal Stochastic Variance Reduced Gradients Algorithm](https://arxiv.org/pdf/1403.4699.pdf).

The code is written in PyTorch. The code is cuda enabled and can be executed on a GPU out of the box.

# Dataset Details
-------------------
- Download the dataset required for the analysis from the following locations:
  - [rcv1](https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/rcv1_train.binary.bz2)
  - [covertype](https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/covtype.libsvm.binary.scale.bz2)
  - [sido0](http://www.causality.inf.ethz.ch/data/sido0_text.zip)

- The datasets need to be uncompressed manually by the user. We couldn't generalize it due to different decompressing utilities exist on different systems.

# Setup instructions
----------------------
- From the root of the directory, execute the following:
  ```sh
  sh ./utils/download_dataset.sh
  ```
- Then, execute the following command to install all the dependencies:
  ```sh
  sh ./install.sh
  ```

# Test scripts
---------------
Scripts are placed in the ./test directory
  - test_dataset_loader.py: Processes the dataset.
  - test_loss.py: Tests if the loss functions return appropriate values.
  - test_optimizers.py: Tests if the optimizers work.

# Result scripts
---------------
Scripts are placed in the ./results directory
  - train_covertype.py: Runs the experiments on Covertype dataset.
  - train_rcv.py: Runs the experiments on Covertype dataset.
  - train_sido.py: Runs the experiments on Covertype dataset.

The scripts generate pickle files with the metrics, such as Number of Non-Zeros, objective gap and the number of effective passes.

# Plot scripts
---------------
The combined plots can be obtained by executing the file plot_from_pkl.py
