import numpy as np
import pandas as pd
import scanpy as sc

np.random.seed(0)


def main():
    # Save the information in another format
    # clusters = pd.read_csv("/usr/xtmp/hyhuang/biopacmap/analysis/clustering/graphclust/clusters.csv")
    sc.settings.verbosity = 2

    # Data file is from here 
    # https://support.10xgenomics.com/single-cell-gene-expression/datasets/1.3.0/1M_neurons
    adata = sc.read_10x_h5('/usr/xtmp/hyhuang/biopacmap/1M_neurons_filtered_gene_bc_matrices_h5.h5')
    sc.pp.recipe_zheng17(adata)
    print(f"1M data has shape: {adata.X.shape}")

    # X = np.copy(adata.X)
    # X = X - X.mean(axis=0)
    # U, s, V = np.linalg.svd(X, full_matrices=False)
    # U[:, np.sum(V,axis=1)<0] *= -1
    # X = np.dot(U, np.diag(s))
    # X = X[:, np.argsort(s)[::-1]][:,:100] # 100 Dimensions
    # np.save('/usr/xtmp/hyhuang/biopacmap/1M_neurons_100pc_npy.npy', X)

    # # load cluster labels
    # # https://github.com/theislab/scanpy_usage/blob/master/170522_visualizing_one_million_cells/results/louvain.csv.gz
    # clusters = pd.read_csv('/usr/xtmp/hyhuang/biopacmap/louvain.csv.gz', header=None).values[:,1].astype(int)
    # np.save('/usr/xtmp/hyhuang/biopacmap/1M_neurons_cluster_npy.npy', clusters)


if __name__ == "__main__":
    main()
