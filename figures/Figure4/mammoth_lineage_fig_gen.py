import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import matplotlib
import json
import pandas as pd
from matplotlib.colors import ListedColormap
# sns.set()
# matplotlib.rc('text', usetex=True)
# matplotlib.rcParams['text.latex.preamble']=[r"\usepackage{amsmath}"]

methods = ['t-SNE', 'LargeVis', 'UMAP', 'TriMap', 'PaCMAP']
results = np.load('./data_extra/pca_mammoth.npy')
tsne = [np.load('../../output_extra/mammoth_t-SNE_perp10.npy')[0],
        np.load('../../output_extra/mammoth_t-SNE_perp15.npy')[0],
        np.load('../../output_extra/mammoth_t-SNE_perp30.npy')[0]]
umap = [np.load('../../output_extra/mammoth_UMAP_perp10.npy')[0],
        np.load('../../output_extra/mammoth_UMAP_perp15.npy')[0],
        np.load('../../output_extra/mammoth_UMAP_perp30.npy')[0]]
trimap = [np.load('../../output_extra/mammoth_TriMap_perp10.npy')[0],
          np.load('../../output_extra/mammoth_TriMap_perp15.npy')[0],
          np.load('../../output_extra/mammoth_TriMap_perp30.npy')[0]]
artsne = np.load('../../output/mammoth_art-SNE.npy')[0]
fa2 = np.load('../../output/mammoth_ForceAtlas2.npy')[0]
PaCMAP = np.load('../../output/mammoth_PaCMAP.npy')[0]
tsne_list = ['10', '15', '30']
umap_list = ['10', '15', '30']
trimap_list = ['10', '15', '30']


def data_prep(dataset='MNIST', size=-1, dim=70):
    if dataset == 'MNIST':
        X = np.load('../../data/mnist_pca.npy', allow_pickle=True).reshape(70000, -1)
        labels = np.load('../../data/mnist_labels.npy', allow_pickle=True)
    elif dataset == 'mammoth':
        with open('../../data/mammoth_3d.json', 'r') as f:
            X = json.load(f)
        X = np.array(X)
        with open('../../data/mammoth_umap.json', 'r') as f:
            labels = json.load(f)
        labels = labels['labels']
        labels = np.array(labels)
        labels = np.sort(labels)
    elif dataset == 'hierarchical_three':
        X = np.load('../../data/hierarchical_threelayer_dataset.npy', allow_pickle=True)
        labels = np.load("../../data/hierarchical_threelayer_label.npy", allow_pickle=True)
    elif dataset == "ercc":
        X = np.load("../../data/zheng_ercc_log_pca.npy", allow_pickle=True)
        labels = np.zeros(10) # no labels
    elif dataset == "monocyte":
        X = np.load("../../data/zheng_monocyte_log_pca.npy", allow_pickle=True)
        labels = np.zeros(10) # no labels
    elif dataset == 'duo4eq':
        X = np.load("../../data/4eq_log_pca.npy", allow_pickle=True)
        labels = np.load("../../data/4eq_labels.npy", allow_pickle=True)
    elif dataset == 'duo8eq':
        X = np.load("../../data/8eq_log_pca.npy", allow_pickle=True)
        labels = np.load("../../data/8eq_labels.npy", allow_pickle=True)
    elif dataset == 'kazer':
        X = np.load("../../data/hiv_70.npy", allow_pickle=True)
        labels = np.load("../../data/kazer_raw_labels.npy", allow_pickle=True)
    elif dataset == 'stuart':
        X = np.load("../../data/seurat_bmnc_rna_70.npy", allow_pickle=True)
        labels = np.load("../../data/stuart_labels.npy", allow_pickle=True)
    elif dataset == 'muraro':
        X = np.load("../../data/muraro_log_pca.npy", allow_pickle=True)
        labels = np.load("../../data/muraro_labels.npy", allow_pickle=True)
    elif dataset == 'kang':
        X = np.load("../../data/kang_log_pca.npy", allow_pickle=True)
        labels = np.load("../../data/kang_labels.npy", allow_pickle=True)
    elif dataset == 'kazerres':
        X = np.load("../../data/kazer_pcares.npy", allow_pickle=True)
        labels = np.load("../../data/kazer_raw_labels.npy", allow_pickle=True)
    elif dataset == 'micebrain':
        X = np.load("../../data/1M_neurons_100pc_npy.npy", allow_pickle=True)
        labels = np.load("../../data/1M_neurons_cluster_npy.npy", allow_pickle=True)
    elif dataset == 'miceembryo':
        X = np.load('../../data/2M_neurons_100pc_npy.npy', allow_pickle=True)
        labels = np.load('../../data/2M_neurons_label.npy', allow_pickle=True)
        labels += 1 # move one forward
    elif dataset == 'lineage':
        X = np.load('../../data/lineage_dataset.npy', allow_pickle=True)
        labels = np.load('../../data/lineage_label.npy', allow_pickle=True)
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


big_title = 24
small_title = 20

# General Figure
fig = plt.figure(figsize=(15, 15.5), dpi=250)
gs = fig.add_gridspec(7, 6, height_ratios=[3, 3, 3, 0.25, 3, 0.25, 3])

# Mammoth
ax3d = fig.add_subplot(gs[0:2, 0:3], projection='3d')
X, y = data_prep('mammoth')

ax3d.scatter(-X[:, 0], X[:, 2],
            X[:, 1], s=1.5, c=y, cmap='viridis')
ax3d.set_title('Original Mammoth')
ax3d.text2D(0, 1, 'a', fontsize=20, weight='bold', ha='center', va='center', transform=ax3d.transAxes)

ax3d.view_init(20, 295)
ax3d.title.set_fontsize(big_title)
plt.draw()

for i in range(3):
    fax = fig.add_subplot(gs[i, 3])
    mam = tsne[i]
    fax.axis('off')
    fax.set_aspect('equal', adjustable='datalim') # set aspect to equal
    fax.scatter(mam[:, 0], mam[:, 1], s=0.02, c=y, cmap='viridis')
    if i != 2:
        fax.set_title(f't-SNE ({tsne_list[i]})')
    else:
        fax.set_title(f't-SNE ({tsne_list[i]})')
    fax.title.set_fontsize(small_title)
    if i == 2:
        fax.title.set_fontweight('bold')
    if i == 0:
        fax.text(-0.1, 1, 'b', fontsize=20, weight='bold', ha='center', va='center', transform=fax.transAxes)

    fax = fig.add_subplot(gs[i, 4])
    mam = umap[i]
    fax.axis('off')
    fax.set_aspect('equal', adjustable='datalim') # set aspect to equal
    fax.scatter(mam[:, 0], mam[:, 1], s=0.02, c=y, cmap='viridis')
    if i != 1:
        fax.set_title(f'UMAP ({umap_list[i]})')
    else:
        fax.set_title(f'UMAP ({umap_list[i]})', fontweight="bold")
    fax.title.set_fontsize(small_title)

    fax = fig.add_subplot(gs[i, 5])
    mam = trimap[i]
    fax.set_aspect('equal', adjustable='datalim') # set aspect to equal
    fax.scatter(mam[:, 0], mam[:, 1], s=0.02, c=y, cmap='viridis')
    fax.axis('off')
    if i != 0:
        fax.set_title(f'TriMap ({trimap_list[i]})')
    else:
        fax.set_title(f'TriMap ({trimap_list[i]})', fontweight="bold")
    fax.title.set_fontsize(small_title)

fax = fig.add_subplot(gs[2, 0])
mam = artsne
fax.set_aspect('equal', adjustable='datalim') # set aspect to equal
fax.scatter(x=mam[:, 0], y=mam[:, 1], s=0.02, c=y, cmap='viridis')
fax.axis('off')
fax.set_title('art-SNE')
fax.title.set_fontsize(small_title)

fax = fig.add_subplot(gs[2, 1])
mam = fa2
fax.set_aspect('equal', adjustable='datalim') # set aspect to equal
fax.scatter(x=mam[:, 0], y=mam[:, 1], s=0.02, c=y, cmap='viridis')
fax.axis('off')
fax.set_title('ForceAtlas2')
fax.title.set_fontsize(small_title)

fax = fig.add_subplot(gs[2, 2])
mam = PaCMAP
fax.set_aspect('equal', adjustable='datalim') # set aspect to equal
fax.scatter(x=mam[:, 0], y=mam[:, 1], s=0.02, c=y, cmap='viridis')
fax.axis('off')
fax.set_title('PaCMAP')
fax.title.set_fontsize(small_title)
# plt.tight_layout()


# Lineage
X = np.load('/usr/xtmp/hyhuang/MNIST/lineage_dataset.npy', allow_pickle=True)
y = np.load('/usr/xtmp/hyhuang/MNIST/lineage_label.npy', allow_pickle=True)
Z_fa = np.load('/home/home1/hh219/Bio_PaCMAP/output/lineage_ForceAtlas2.npy', allow_pickle=True)[0]
Z_ts = np.load('/home/home1/hh219/Bio_PaCMAP/output/lineage_t-SNE.npy', allow_pickle=True)[0]
Z_arts = np.load('/home/home1/hh219/Bio_PaCMAP/output/lineage_art-SNE.npy', allow_pickle=True)[0]
Z_um = np.load('/home/home1/hh219/Bio_PaCMAP/output/lineage_UMAP.npy', allow_pickle=True)[0]
Z_tr = np.load('/home/home1/hh219/Bio_PaCMAP/output/lineage_TriMap.npy', allow_pickle=True)[0]
Z_pc = np.load('/home/home1/hh219/Bio_PaCMAP/output/lineage_PaCMAP.npy', allow_pickle=True)[0]
Zs = [Z_fa, Z_ts, Z_arts, Z_um, Z_tr, Z_pc]
names = ['ForceAtlas2', 't-SNE', 'art-SNE', 'UMAP', 'TriMAP', 'PaCMAP']
# fig, ax = plt.subplots(2, 3, figsize=(12, 8))
# ax = ax.flatten()
for i in range(6):
    fax = fig.add_subplot(gs[4, i])
    fax.set_aspect('equal', adjustable='datalim') # set aspect to equal
    fax.scatter(Zs[i][:, 0], Zs[i][:, 1], c=y, cmap='viridis', s=0.06)
    fax.axis('off')
    fax.set_title(names[i], fontsize=small_title)
    if i == 0:
        fax.text(-0.1, 1, 'c', fontsize=20, weight='bold', ha='center', va='center', transform=fax.transAxes)

# Hierarchical
X = np.load('/usr/xtmp/hyhuang/MNIST/hierarchical_threelayer_dataset.npy', allow_pickle=True)
y = np.load("/usr/xtmp/hyhuang/MNIST/hierarchical_threelayer_label.npy", allow_pickle=True)
Z_fa = np.load('/home/home1/hh219/Bio_PaCMAP/output/hierarchical_three_ForceAtlas2.npy', allow_pickle=True)[0]
Z_ts = np.load('/home/home1/hh219/Bio_PaCMAP/output/hierarchical_three_t-SNE.npy', allow_pickle=True)[0]
Z_arts = np.load('/home/home1/hh219/Bio_PaCMAP/output/hierarchical_three_art-SNE.npy', allow_pickle=True)[0]
Z_um = np.load('/home/home1/hh219/Bio_PaCMAP/output/hierarchical_three_UMAP.npy', allow_pickle=True)[0]
Z_tr = np.load('/home/home1/hh219/Bio_PaCMAP/output/hierarchical_three_TriMap.npy', allow_pickle=True)[0]
Z_pc = np.load('/home/home1/hh219/Bio_PaCMAP/output/hierarchical_three_PaCMAP.npy', allow_pickle=True)[0]

a = sns.color_palette("Reds", n_colors=10).as_hex()
b = sns.color_palette("Blues", n_colors=10).as_hex()
c = sns.color_palette("Greens", n_colors=10).as_hex()
d = sns.color_palette("Greys", n_colors=10).as_hex()
e = sns.color_palette("copper_r", n_colors=10).as_hex()
a = a[5:]
b = b[5:]
c = c[5:]
d = d[5:]
e = e[5:]
cm = a + b + c + d + e
second_layer_clist = []
for i in range(25):
    for j in range(5):
        second_layer_clist.append(cm[i])
second_cmap = ListedColormap(second_layer_clist)

Zs = [Z_fa, Z_ts, Z_arts, Z_um, Z_tr, Z_pc]
# names = ['ForceAtlas2', 't-SNE', 'art-SNE', 'UMAP', 'TriMAP', 'PaCMAP']
for i in range(6):
    fax = fig.add_subplot(gs[6, i])
    fax.set_aspect('equal', adjustable='datalim') # set aspect to equal
    fax.scatter(Zs[i][:, 0], Zs[i][:, 1], c=y, cmap=second_cmap, s=0.06)
    fax.axis('off')
    fax.set_aspect('equal', adjustable='datalim') # set aspect to equal
    fax.set_title(names[i], fontsize=small_title)
    if i == 0:
        fax.text(-0.1, 1, 'd', fontsize=20, weight='bold', ha='center', va='center', transform=fax.transAxes)

# plt.tight_layout()

plt.savefig('fig_combined_2203.png')
