import sys
import time

import numpy as np
import faiss
from faiss.contrib.ondisk import merge_ondisk
from python.utils.dataset import BigANNVectorDataSet, BigANNNeighborDataSet
from python.utils.utils import recall_at_r

from faiss import omp_set_num_threads

stage = int(sys.argv[1])

tmpdir = '/home/ec2-user/tmp/'
data = "/home/ec2-user/data/BIGANN-base.1B.u8bin"
queries = "/home/ec2-user/data/BIGANN-query.public.10K.u8bin"
gt = "/home/ec2-user/data/BIGANN-public_query_gt100.bin"

# data = "/home/ec2-user/data/sift-128-euclidean.hdf5"
# queries = "/home/ec2-user/data/sift-128-euclidean.hdf5"
# gt = "/home/ec2-user/data/sift-128-euclidean.hdf5"

nlist = 65_536
train_count = 2_000_000
index_count = 1_000_000_000
batch_size = 10_000_000

def run_experiment(stage):

    omp_set_num_threads(12)

    index_dataset = BigANNVectorDataSet(data)
    #index_dataset = HDF5DataSet(data, Context.INDEX)

    gt_dataset = BigANNNeighborDataSet(gt)
    queries_dataset = BigANNVectorDataSet(queries)
    #gt_dataset = HDF5DataSet(gt, Context.NEIGHBORS)
    #queries_dataset = HDF5DataSet(queries, Context.QUERY)

    time_file = "metrics/{}_time.txt".format(stage)

    start = None
    end = None
    if stage == 0:
        index_description = "IVF{}(HNSW16_SQfp16),Flat".format(nlist)
        print("Training Index with descrption\"{}\" and training points={}".format(index_description, train_count))
        print("Reading dataset...")
        training_vectors = index_dataset.read(nlist)

        index = faiss.index_factory(training_vectors.shape[1], index_description)
        print("Training index")
        start = time.time()
        index.train(training_vectors)
        end = time.time()
        print("Training complete")
        print("write " + tmpdir + "trained.index")
        faiss.write_index(index, tmpdir + "trained.index")

    if stage == 1:
        start = time.time()
        batches = index_count // batch_size
        curr_start = 0
        for bno in range(batches):
            xb = index_dataset.read_batch(batch_size)
            index = faiss.read_index(tmpdir + "trained.index")
            print("adding vectors %d:%d" % (curr_start, curr_start + batch_size))
            index.add_with_ids(xb, np.arange(curr_start, curr_start + batch_size))
            print("write " + tmpdir + "block_%d.index" % bno)
            faiss.write_index(index, tmpdir + "block_%d.index" % bno)
            curr_start += batch_size
        end = time.time()

    if stage == 2:
        print('loading trained index')
        # construct the output index
        index = faiss.read_index(tmpdir + "trained.index")

        block_fnames = [
            tmpdir + "block_%d.index" % bno
            for bno in range(index_count // batch_size)
        ]

        start = time.time()
        merge_ondisk(index, block_fnames, tmpdir + "merged_index.ivfdata")
        end = time.time()
        print("write " + tmpdir + "populated.index")
        faiss.write_index(index, tmpdir + "populated.index")

    if stage == 3:
        # perform a search from disk
        print("read " + tmpdir + "populated.index")
        index = faiss.read_index(tmpdir + "populated.index")
        index.nprobe = 16

        # load query vectors and ground-truth
        start = time.time()
        xq = queries_dataset.read(queries_dataset.size())
        D, I = index.search(xq, 10)
        end = time.time()
        print(recall_at_r(I, gt_dataset, 10, 10, 10000))

    with open(time_file, 'w') as f:
        t = "{}\n".format((end - start))
        f.write(t)
