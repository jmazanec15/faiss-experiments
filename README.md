# faiss experimental

Project containing experimental client code using the ** faiss ** library in order to test out different concepts.


## Run python code 

Prereqs:
1. Python 3.8 environment

```bash

cmake -Bbuild .
make -C build -j faiss swigfaiss

(cd build/external/faiss/faiss/python && python3 setup.py build)
PYTHONPATH="$(ls -d ./build/external/faiss/faiss/python/build/lib*/)"

```