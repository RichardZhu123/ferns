import h5py
import matplotlib.pyplot as plt
import numpy as np
import random
import time
from ferns import VectorTree
from tqdm import tqdm

def run_on_bench(data):
    """ data should be an np array of shape (num_vectors, dim) """
    retrieval_times = [] # retrieval times for each trial
    proportion_traversed = [] # avg. proportion of graph traversed per num_vectors item
    recalls = [] # avg. recall per num_vectors item
    max_tree_depths = [] # max tree depth per tree
    tree = VectorTree()

    num_vectors = data.shape[0]
    num_inserted = 0

    # breakpoints to evaluate runtimes
    bkpts = [num_vectors//1e3, num_vectors//1e2, num_vectors//10, num_vectors]

    for vec in tqdm(data):
        tree.insert(vec)
        num_inserted += 1

        if num_inserted in bkpts:
            # create an empty list to store retrieval times for this num_vector configuration
            retrieval_times_for_selected_vector = []
            num_visited_for_selected_vector = []
            recall_for_selected_vector = []
                
            # for each retrieval
            for idx in random.sample(range(num_inserted), min(num_inserted, 1000)):
                query = data[idx]
                tic = time.time()
                vector, num_visited = tree.retrieve_nearest(query)
                retrieval_times_for_selected_vector.append(time.time() - tic)
                num_visited_for_selected_vector.append(num_visited)
                recall_for_selected_vector.append(np.array_equal(np.array(vector), np.array(query)))
                            
            retrieval_times.append(sum(retrieval_times_for_selected_vector) / len(retrieval_times_for_selected_vector))
            proportion_traversed.append(sum(num_visited_for_selected_vector) / len(num_visited_for_selected_vector) / num_inserted)
            recalls.append(sum(recall_for_selected_vector) / len(recall_for_selected_vector))
            max_tree_depths.append(tree.max_depth)

    print("Retrieval times:", retrieval_times)
    print("Proportion of vectors traversed:", proportion_traversed)
    print("Recall rate:", recalls)
    print("Max tree depths:", max_tree_depths)
    
    return bkpts, retrieval_times

def main():
    ls_ds = ['fashion-mnist-784-euclidean', 'mnist-784-euclidean', 'sift-128-euclidean']
    for str_ds in ls_ds:
        with h5py.File(f"data/{str_ds}.hdf5", 'r') as f:
            data = f['train'][:]
            num_vectors, retrieval_times = run_on_bench(data)
            plt.plot(num_vectors, retrieval_times, label=str_ds)

    plt.xlabel('# of vectors in database')
    plt.ylabel('Average retrieval time on vector database benchmarks')
    plt.legend(title='Benchmark')
    plt.xscale('log')
    plt.tight_layout()
    plt.savefig('res/ann-bench.png')
    plt.show()

if __name__ == "__main__":
    main()