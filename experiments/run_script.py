import umap
import trimap
import json
import openTSNE
import pacmap
import fitsne
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from fa2 import ForceAtlas2 as FA2
from time import time
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
from sklearn.neighbors import NearestNeighbors
from utils import make_adj_sklearn, make_adj_annoy

def data_prep(dataset='MNIST', size=-1, dim=70):
    if dataset == 'MNIST':
        X = np.load('/usr/xtmp/hyhuang/MNIST/mnist_pca.npy', allow_pickle=True).reshape(70000, -1)
        labels = np.load('/usr/xtmp/hyhuang/MNIST/mnist_labels.npy', allow_pickle=True)
    elif dataset == 'FMNIST':
        X = np.load('/usr/xtmp/hyhuang/MNIST/fmnist_images.npy', allow_pickle=True).reshape(70000, 28*28)
        labels = np.load('/usr/xtmp/hyhuang/MNIST/fmnist_labels.npy', allow_pickle=True)
    elif dataset == 'mammoth':
        with open('/usr/xtmp/hyhuang/MNIST/mammoth_3d.json', 'r') as f:
            X = json.load(f)
        X = np.array(X)
        with open('/usr/xtmp/hyhuang/MNIST/mammoth_umap.json', 'r') as f:
            labels = json.load(f)
        labels = labels['labels']
        labels = np.array(labels)
        labels = np.sort(labels)
    elif dataset == 'mammoth_50k':
        with open('/usr/xtmp/hyhuang/MNIST/mammoth_3d_50k.json', 'r') as f:
            X = json.load(f)
        X = np.array(X)
        labels = np.zeros(10)
    elif dataset == 'Mouse_scRNA':
        data = pd.read_csv('/usr/xtmp/hyhuang/GSE93374_Merged_all_020816_BatchCorrected_LNtransformed_doubletsremoved_Data.txt', sep='\t')
        X = data.to_numpy()
        labels = pd.read_csv('/usr/xtmp/hyhuang/GSE93374_cell_metadata.txt', sep='\t')
    elif dataset == 'hierarchical':
        X = np.load('/usr/xtmp/hyhuang/MNIST/hierarchical_dataset.npy', allow_pickle=True)
        labels = np.load('/usr/xtmp/hyhuang/MNIST/hierarchical_label.npy', allow_pickle=True)
    elif dataset == 'hierarchical_three':
        X = np.load('/usr/xtmp/hyhuang/MNIST/hierarchical_threelayer_dataset.npy', allow_pickle=True)
        labels = np.load("/usr/xtmp/hyhuang/MNIST/hierarchical_threelayer_label.npy", allow_pickle=True)
    elif dataset == "ercc":
        X = np.load("./scRNA-DR2020/data/zheng_ercc_log_pca.npy", allow_pickle=True)
        labels = np.zeros(10) # no labels
    elif dataset == "monocyte":
        X = np.load("./scRNA-DR2020/data/zheng_monocyte_log_pca.npy", allow_pickle=True)
        labels = np.zeros(10) # no labels
    elif dataset == 'duo4eq':
        X = np.load("./scRNA-DR2020/data/4eq_log_pca.npy", allow_pickle=True)
        labels = np.load("./scRNA-DR2020/data/4eq_labels.npy", allow_pickle=True)
    elif dataset == 'duo8eq':
        X = np.load("./scRNA-DR2020/data/8eq_log_pca.npy", allow_pickle=True)
        labels = np.load("./scRNA-DR2020/data/8eq_labels.npy", allow_pickle=True)
    elif dataset == 'kazer':
        X = np.load("./scRNA-DR2020/data/hiv_70.npy", allow_pickle=True)
        labels = np.load("./scRNA-DR2020/data/kazer_raw_labels.npy", allow_pickle=True)
    elif dataset == 'stuart':
        X = np.load("./scRNA-DR2020/data/seurat_bmnc_rna_70.npy", allow_pickle=True)
        labels = np.load("./scRNA-DR2020/data/stuart_labels.npy", allow_pickle=True)
    elif dataset == 'muraro':
        X = np.load("./scRNA-DR2020/data/muraro_log_pca.npy", allow_pickle=True)
        labels = np.load("./scRNA-DR2020/data/muraro_labels.npy", allow_pickle=True)
    elif dataset == 'kang':
        X = np.load("./scRNA-DR2020/data/kang_log_pca.npy", allow_pickle=True)
        labels = np.load("./scRNA-DR2020/data/kang_labels.npy", allow_pickle=True)
    elif dataset == 'kazerres':
        X = np.load("./scRNA-DR2020/data/kazer_pcares.npy", allow_pickle=True)
        labels = np.load("./scRNA-DR2020/data/kazer_raw_labels.npy", allow_pickle=True)
    elif dataset == 'micebrain':
        X = np.load("/usr/xtmp/hyhuang/biopacmap/1M_neurons_100pc_npy.npy", allow_pickle=True)
        labels = np.load("/usr/xtmp/hyhuang/biopacmap/1M_neurons_cluster_npy.npy", allow_pickle=True)
    elif dataset == 'miceembryo':
        X = np.load('/usr/xtmp/hyhuang/biopacmap/2M_neurons_100pc_npy.npy', allow_pickle=True)
        labels = np.load('/usr/xtmp/hyhuang/biopacmap/2M_neurons_label.npy', allow_pickle=True)
        labels += 1 # move one forward
    elif dataset == 'lineage':
        X = np.load('/usr/xtmp/hyhuang/MNIST/lineage_dataset.npy', allow_pickle=True)
        labels = np.load('/usr/xtmp/hyhuang/MNIST/lineage_label.npy', allow_pickle=True)
    else:
        print('Unsupported dataset')
        assert(False)
    if size != -1:
        X = X[:size, :dim]
        labels = labels[:size]
    else:
        X = X[:, :dim]
    X = X.copy(order='C')
    return X, labels


def experiment_five(X, method='PaCMAP', **kwargs):
    X_lows, running_times = [], []
    for i in range(5):
        X_low, running_time = experiment(X, method, **kwargs)
        X_lows.append(X_low)
        running_times.append(running_time)
    X_lows = np.array(X_lows)
    running_times = np.array(running_times)
    return X_lows, running_times


def experiment(X, method='PaCMAP', **kwargs):
    start_time = time()
    if method == 'PaCMAP':
        X_low = transform_by_PACMAP(X, **kwargs)
    elif method == 'UMAP':
        X_low = transform_by_UMAP(X, **kwargs)
    elif method == 'TriMap':
        X_low = transform_by_TRIMAP(X, **kwargs)
    elif method == 't-SNE':
        X_low = transform_by_TSNE(X, **kwargs)
    elif method == 'art-SNE':
        X_low = transform_by_ARTSNE(X)
    elif method == 'ForceAtlas2':
        X_low = transform_by_FA2(X, **kwargs)
    elif method == 'ForceAtlas2_ANNOY':
        X_low = transform_by_FA2_annoy(X, **kwargs)
    elif method == 'PCA':
        X_low = transform_by_PCA(X, **kwargs)
    else:
        print("Incorrect method specified")
        raise ValueError
    running_time = time() - start_time
    return X_low, running_time


def transform_by_PCA(X, n_components=2):
    # Initialize the embedding.
    p = PCA(n_components=n_components)
    X_low = p.fit_transform(X)
    return X_low


def transform_by_FA2(X, n_neighbors=15):
    A = make_adj_sklearn(X, n_neighbors=n_neighbors)

    # Initialize the embedding.
    p = PCA(n_components=2)
    X = p.fit_transform(X)
    X_init = X[:, :2]
    X_init *= 10000/np.std(X_init)

    # Perform the extraction
    f = FA2(verbose=False)
    X_low = f.forceatlas2(A, X_init, 100)
    X_low = np.array(X_low)
    return X_low


def transform_by_FA2_annoy(X, n_neighbors=5):
    A, _ = make_adj_annoy(X, n_neighbors=n_neighbors)

    # Initialize the embedding.
    p = PCA(n_components=2)
    X = p.fit_transform(X)
    X_init = X[:, :2]
    X_init *= 10000/np.std(X_init)

    # Perform the extraction
    f = FA2(verbose=False)
    X_low = f.forceatlas2(A, X_init, 100)
    X_low = np.array(X_low)
    return X_low


def transform_by_TSNE(X, **kwargs):
    tfmr = openTSNE.TSNE(**kwargs)
    X_low = tfmr.fit(X)
    return X_low


def transform_by_UMAP(X, **kwargs):
    tfmr = umap.UMAP(**kwargs)
    X_low = tfmr.fit_transform(X)
    return X_low


def transform_by_TRIMAP(X, **kwargs):
    tfmr = trimap.TRIMAP(verbose=False, **kwargs)
    X_low = tfmr.fit_transform(X)
    return X_low


def transform_by_PACMAP(X, **kwargs):
    tfmr = pacmap.PaCMAP(n_neighbors=10, **kwargs) # Fix the dimensions
    X_low = tfmr.fit_transform(X)
    return X_low


def transform_by_ARTSNE(X):
    pcaInit = PCA(n_components=2).fit_transform(X)
    pcaInit = pcaInit / np.std(pcaInit[:,0]) * 0.0001
    X_low = fitsne.FItSNE(X, perplexity_list=[30, int(X.shape[0]/100)], 
                initialization=pcaInit, learning_rate=X.shape[0]/12)
    return X_low


if __name__ == '__main__':
    # do experiment
    pass
