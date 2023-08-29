# faiss experimental

Project containing experimental client code using the ** faiss ** library in order to test out different concepts.


## Setup AL2023

Install prereqs:
```bash
sudo yum install gcc-c++ tmux git zlib-devel openblas-devel gfortran -y

```

Get conda from https://www.anaconda.com/download/

Create 3.8 Conda environment:
```bash
conda create -n knn-perf python=3.8
conda activate knn-perf
```

Install python reqs:
```bash
pip install numpy cmake==3.23.3 swig h5py psutil
```

## Run python code

```bash

cmake -Bbuild .
make -C build -j faiss swigfaiss

(cd build/external/faiss/faiss/python && python3 setup.py build)
export PYTHONPATH="$(ls -d `pwd`/build/external/faiss/faiss/python/build/lib*/):`pwd`/"

python python/disk_experiment.py
```