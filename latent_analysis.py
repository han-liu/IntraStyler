import os
import os.path as osp
import pickle
import numpy as np
import argparse
from sklearn.decomposition import PCA
from sklearn.cluster import MiniBatchKMeans, KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_samples, silhouette_score
import faiss
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D



parser = argparse.ArgumentParser()
parser.add_argument('--dir', help='directory of saved embeddings')
args = parser.parse_args()


def load_styles(dirc: str):    
    with open(osp.join(dirc, 'style_vectors.npz'), 'rb') as f:
        styles = pickle.load(f)
    with open(osp.join(dirc, 'sites.npz'), 'rb') as f:
        sites = pickle.load(f)
    with open(osp.join(dirc, 'subject_ids.npz'), 'rb') as f:
        subject_ids = pickle.load(f)
    return np.array(styles), np.array(subject_ids), np.array(sites)


def run_PCA(feats, ids, sites, weights=None):
    for n_comp in range(2, 5):
        pca = PCA(n_components=n_comp)
        pca.fit(feats)
        feats_t = pca.transform(feats)
        var_ratio = pca.explained_variance_ratio_
        if var_ratio.sum() > 0.99:  
            print(f'Feature matrix dimension: {feats.shape}')
            print(f'Feature matrix dimension (after transformation): {feats_t.shape}')
            print(f'{n_comp} principal components have explained {var_ratio.sum()*100:.2f}% variance.')
            singular_vals = pca.singular_values_
            mean = pca.mean_
            eigenvectors = pca.components_
            recon = mean + np.dot(feats_t, eigenvectors)

            # create a new style based on eigenvectors
            aug_style = None
            if weights is not None:
                weights /= np.array(weights).sum()
                new_style = np.dot(weights, eigenvectors)
                aug_style = mean + new_style
                norm = np.linalg.norm(aug_style)
                aug_style /= norm

            return pca, mean, eigenvectors, aug_style


def visualize_PCA(feats, pca, sites=None, aug=None):
    if aug is not None:
        aug = aug.reshape(-1, feats.shape[1])
        feats = np.concatenate((feats, aug), 0)
        sites = np.append(sites, 'syn')

    feats_t = pca.transform(feats)
    # pc1, pc2, pc3 = feats_t[:, 0], feats_t[:, 1], feats_t[:, 2]  # 3D
    pc1, pc2 = feats_t[:, 0], feats_t[:, 1]  #2D
    fig = plt.figure(figsize=(10, 10))
    # ax = fig.add_subplot(111, projection='3d')  # 3D
    ax = fig.add_subplot(111)  # 2D
    if sites is not None:
        colors = plt.cm.Set2(np.linspace(0, 1, len(set(sites))))
        for site, color in zip(set(sites), colors):
            print(site, color)
            ix = np.where(sites == site)
            # ax.scatter(pc1[ix], pc2[ix], pc3[ix], c=[color], label=site)  # 3D
            if site != -1:
                ax.scatter(pc1[ix], pc2[ix], c=[color], label=site, s=30) # 2D
            else:
                ax.scatter(pc1[ix], pc2[ix], c='gold', marker='*', label=site, s=120, edgecolors='black', linewidths=0.8) # 2D
    else:
        ax.scatter(pc1, pc2, pc3)
    
    ax.set_xlabel('PC1', fontsize=8)
    ax.set_ylabel('PC2', fontsize=8)
    ax.set_xticks([])
    ax.set_yticks([])
    # ax.set_zlabel('PC3')
    # plt.legend()
    # plt.title('Style Embedding Space')
    plt.show()
            

def active_selection(feats:np.array, num_samples:int, local_neighbor:int=30):
    """ TypiClust is used for diversity-based style selection
    paper link: https://arxiv.org/pdf/2202.02794.pdf

    feats: n samples x 256-dim style
    """

    def get_nn(feats, num_neighbors):
        # calculates nearest neighbors on GPU
        d = feats.shape[1]
        feats = feats.astype(np.float32)
        cpu_index = faiss.IndexFlatL2(d)
        # cpu_index = faiss.IndexFlatIP(d)
        # gpu_index = faiss.index_cpu_to_all_gpus(cpu_index)
        # gpu_index.add(feats)  # add vectors to the index
        cpu_index.add(feats)
        distances, indices = cpu_index.search(feats, num_neighbors + 1)
        # 0 index is the same sample, dropping it
        return distances[:, 1:], indices[:, 1:]

    def get_mean_nn_dist(feats, num_neighbors, return_indices=False):
        distances, indices = get_nn(feats, num_neighbors)
        mean_distance = distances.mean(axis=1)
        if return_indices:
            return mean_distance, indices
        return mean_distance

    def calculate_typicality(feats, num_neighbors):
        mean_distance = get_mean_nn_dist(feats, num_neighbors)
        # low distance to NN is high density
        typicality = 1 / (mean_distance + 1e-5)
        return typicality

    def kmeans(feats, num_clusters):
        sc = []  # silhouette scores
        init_num_clusters = 5
        max_num_clusters = 20 #15
        for i in range(init_num_clusters, max_num_clusters+1):
            km = KMeans(n_clusters=i)
            km.fit(feats)
            silhouette_avg = silhouette_score(feats, km.labels_)
            sc.append(silhouette_avg)

        optimal_clusters = init_num_clusters + np.array(sc).argmax()
        print('optimal number of clusters: ', optimal_clusters)
        print('scores: ', sc)
        num_clusters = optimal_clusters
        # breakpoint()

        if num_clusters <= 50:
            # agg = AgglomerativeClustering(affinity='cosine', linkage='average', n_clusters=num_clusters)
            # agg.fit_predict(feats)
            km = KMeans(n_clusters=num_clusters)
            km.fit_predict(feats)
        else:
            km = MiniBatchKMeans(n_clusters=num_clusters, init='k-means++', batch_size=5000)
            km.fit_predict(feats)
        return km.labels_    
        # return agg.labels_    

    selected, selected_indices = [], []

    cluster_labels = kmeans(feats, num_samples)
    cluster_ids, cluster_sizes = np.unique(cluster_labels, return_counts=True)
    indices = np.arange(feats.shape[0])
    for cluster_id in cluster_ids:
        cluster_feats = feats[cluster_labels==cluster_id]
        cluster_indices = indices[cluster_labels==cluster_id]
        typicality = calculate_typicality(cluster_feats, min(local_neighbor, len(cluster_feats) // 2))
        typi_feat = cluster_feats[typicality.argmax(), :]
        selected.append(typi_feat)
        selected_indices.append(cluster_indices[typicality.argmax()])

    cluster_labels[selected_indices] = -1
    print(cluster_labels)

    return cluster_labels, selected, selected_indices


def shape_analysis():
    pass


def slerp(v1, v2, t, verbose=False):
    v1 /= np.linalg.norm(v1)
    v2 /= np.linalg.norm(v2)

    dot = np.dot(v1, v2)
    # if dot < 0.0:
    #     v2 = -v2
    #     dot = -dot
    dot = np.clip(dot, -1.0, 1.0)

    theta_0 = np.arccos(dot)
    sin_theta_0 = np.sin(theta_0)

    if sin_theta_0 < 1e-6:
        return (1.0 - t) * v1 + t * v2

    theta_t = theta_0 * t  # angle between v1 and the output
    sin_theta_t = np.sin(theta_t)  # compute sine of theta_t

    s1 = np.sin((1.0 - t) * theta_0) / sin_theta_0
    s2 = sin_theta_t / sin_theta_0
    output = s1 * v1 + s2 * v2

    if verbose:
        print(f'norm of out: {np.linalg.norm(output)}')
        print(f'angle between v1  and v2 : {np.arccos(np.dot(v1, v2)) / np.pi * 180:.2f}')
        print(f'angle between v1  and out: {np.arccos(np.dot(v1, output)) / np.pi * 180:.2f}')
        print(f'angle between out and v1 : {np.arccos(np.dot(output, v2)) / np.pi * 180:.2f}')
    return output


def main():
    feats, ids, sites = load_styles(args.dir)
    pca, mu, egv, _ = run_PCA(feats, ids, sites)
    feats_t = pca.transform(feats)
    sites, selected, selected_indices = active_selection(feats, 6)
    print(ids)
    print(ids[selected_indices])
    visualize_PCA(feats, pca, sites)

    # etz_175 = feats[ids == 'etz_175']
    # ldn_144 = feats[ids == 'ldn_144']
    # ldn_145 = feats[ids == 'ldn_145']
    # ukm_66 = feats[ids == 'ukm_66']
    # ukm_106 = feats[ids == 'ukm_106']
    # ukm_107 = feats[ids == 'ukm_107']
    # breakpoint()

    # output = slerp(etz_175.reshape(-1), ukm_66.reshape(-1), t=0.5, verbose=True)
    # output = slerp(etz_175.reshape(-1), ukm_106.reshape(-1), t=0.5, verbose=True)
    # output = slerp(ukm_66.reshape(-1), ukm_106.reshape(-1), t=0.5, verbose=True)
    # output = slerp(etz_175.reshape(-1), ldn_145.reshape(-1), t=0.5, verbose=True)
    # output = slerp(ldn_144.reshape(-1), ldn_145.reshape(-1), t=0.5, verbose=True)

    # v1 = np.array([2.0, 0.0, 0.8])
    # v2 = np.array([0.0, 0.4, 0.6])
    # t = 0.1
    # output = slerp(v1, v2, t, verbose=True)




if __name__ == "__main__":
    main()