#! /bin/bash

mkdir -p ../data/rcv1 ./data/covertype ./data/sido0
echo 'Created directories'

# rcv1
wget -q -O '../data/rcv1/rcv1_train.binary.bz2' 'https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/rcv1_train.binary.bz2'
echo 'Downloaded rcv1'

# covertype
wget -q -O '../data/covertype/covtype.libsvm.binary.bz2' 'https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/covtype.libsvm.binary.bz2'
echo 'Downloaded covertype'

# sido0
wget -q -O '../data/sido0/sido0_text.zip' 'http://www.causality.inf.ethz.ch/data/sido0_text.zip'
echo 'Downloaded sido0'

echo 'All data downloaded'
