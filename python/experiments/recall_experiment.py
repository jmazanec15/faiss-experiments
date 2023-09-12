from python.utils.dataset import HDF5DataSet, Context
from python.utils.utils import recall_at_r

from faiss import omp_set_num_threads, IndexPQ

"""
Light weight experiment to measure recall for different configurations
"""


def run_experiment(stage):

    data = "/home/ec2-user/data/sift-128-euclidean.hdf5"
    queries = "/home/ec2-user/data/sift-128-euclidean.hdf5"
    gt = "/home/ec2-user/data/sift-128-euclidean.hdf5"

    train_count = 65536
    index_count = 1_000_000

    dimension = 128
    M = 64 # number of subvectors to split vector into
    nbits = 12

    omp_set_num_threads(12)

    index_dataset = HDF5DataSet(data, Context.INDEX)
    gt_dataset = HDF5DataSet(gt, Context.NEIGHBORS)
    queries_dataset = HDF5DataSet(queries, Context.QUERY)

    index = IndexPQ(dimension, M, nbits)
    d = index_dataset.read(train_count)
    print("Training")
    index.train(d)
    print("Training Complete")
    index_dataset.reset()
    d = index_dataset.read(index_count)
    print("Indexing")
    index.add(d)
    print("Indexing Complete")

    # load query vectors and ground-truth
    xq = queries_dataset.read(queries_dataset.size())
    print("Querying")
    D, I = index.search(xq, queries_dataset.size())
    print(recall_at_r(I, gt_dataset, 10, 10, queries_dataset.size()))
