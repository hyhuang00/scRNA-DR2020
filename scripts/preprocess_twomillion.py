import numpy as np
import pandas as pd
import scanpy as sc
from scipy.io import mmread
from scipy import sparse
import anndata

np.random.seed(0)


def main():
    # Save the information in another format
    sc.settings.verbosity = 2

    # Data file is from here 
    # https://support.10xgenomics.com/single-cell-gene-expression/datasets/1.3.0/1M_neurons
    counts = mmread('/usr/xtmp/hyhuang/biopacmap/gene_count.txt').transpose()   # 38 min!!
    counts = sparse.csr_matrix(counts)
    adata = anndata.AnnData(counts)
    print(f"2M data has shape: {adata.X.shape}")


    # sc.pp.recipe_zheng17(adata, n_top_genes=2000)  # 2 min

    # X = np.copy(adata.X)
    # X = X - X.mean(axis=0)
    # # Note: svd took extra ~60Gb RAM. scipy.sparse.svds would probably be better here.
    # # I did not want to use randomized algorithms for this paper.
    # U, s, V = np.linalg.svd(X, full_matrices=False) # 1 min 46s
    # U[:, np.sum(V,axis=1)<0] *= -1
    # X = np.dot(U, np.diag(s))
    # X = X[:, np.argsort(s)[::-1]][:,:100]
    # np.save('/usr/xtmp/hyhuang/biopacmap/2M_neurons_100pc_npy.npy', X)

    # load cluster labels
    # https://github.com/theislab/scanpy_usage/blob/master/170522_visualizing_one_million_cells/results/louvain.csv.gz
    # clusters = pd.read_csv('/usr/xtmp/hyhuang/biopacmap/louvain.csv.gz', header=None).values[:,1].astype(int)
    # np.save('/usr/xtmp/hyhuang/biopacmap/1M_neurons_cluster_npy.npy', clusters)


if __name__ == "__main__":
    main()
