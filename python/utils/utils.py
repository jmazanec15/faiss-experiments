

def recall_at_r(results, neighbor_dataset, r, k, query_count):
    """
    Calculates the recall@R for a set of queries against a ground truth nearest
    neighbor set
    Args:
        results: 2D list containing ids of results returned by OpenSearch.
        results[i][j] i refers to query, j refers to
            result in the query
        neighbor_dataset: 2D dataset containing ids of the true nearest
        neighbors for a set of queries
        r: number of top results to check if they are in the ground truth k-NN
        set.
        k: k value for the query
        query_count: number of queries
    Returns:
        Recall at R
    """
    correct = 0.0
    total_num_of_results = 0
    for query in range(query_count):
        true_neighbors = neighbor_dataset.read(1)
        if true_neighbors is None:
            break
        true_neighbors_set = set(true_neighbors[0][:k])
        true_neighbors_set.discard(-1)
        min_r = min(r, len(true_neighbors_set))
        total_num_of_results += min_r
        for j in range(min_r):
            if results[query][j] in true_neighbors_set:
                correct += 1.0

    return correct / total_num_of_results
