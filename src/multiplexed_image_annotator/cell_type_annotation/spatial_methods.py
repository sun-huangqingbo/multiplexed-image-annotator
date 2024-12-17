import os
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import SpectralClustering, HDBSCAN

import numpy as np
import pickle


def _neighborhood_analysis(self, n_neighbors=10, cell_types=None, integrate=False, normalize=True, batch_id=None, result_dir=None):
    colors = ["#FF0000", "#00FF00", "#0000FF", "#FFFF00", "#00FFFF", "#FF00FF", "#FFA500",
            "#800080", "#FFC0CB", "#008080", "#32CD32", "#4B0082", "#808000", "#800000",
            "#000080", "#FFD700", "#EE82EE", "#C0C0C0"]
    colors = {k: colors[i] for i, k in enumerate(cell_types)}
    
    if integrate:
        neighborhood = np.zeros((len(cell_types), len(cell_types)))
        for i, key in enumerate(self.preprocessor.cell_pos_dict.keys()):
            coordinates = self.preprocessor.cell_pos_dict[key]
            # to array
            coordinates = [[np.mean(coordinates[k][0]), np.mean(coordinates[k][1])] for k in sorted(coordinates.keys())]
            assert len(coordinates) == len(self.annotations[i])
            # fit the nearest neighbors
            nbrs = NearestNeighbors(n_neighbors=n_neighbors, algorithm='ball_tree').fit(coordinates)
            for j in range(len(self.annotations[i])):
                indices = nbrs.kneighbors([coordinates[j]], return_distance=False)[0]
                for k in indices[1:]:
                    neighborhood[cell_types.index(self.annotations[i][j]), cell_types.index(self.annotations[i][k])] += 1
        # normalize
        if normalize:
            for i in range(len(neighborhood)):
                if neighborhood[i].sum() > 0:
                    neighborhood[i] /= neighborhood[i].sum()
        # plot
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_title("Integrated neighborhood analysis")
        labels = cell_types
        sns.heatmap(neighborhood, xticklabels=labels, yticklabels=labels, cmap="vlag", linewidth=.5)
        plt.xticks(rotation=60)
        plt.tight_layout()
        f = os.path.join(result_dir, f"{batch_id}_integrated_neighborhood.png")
        plt.savefig(f)
        plt.close()

        # pickle the neighborhood
        f = os.path.join(result_dir, f"{batch_id}_integrated_neighborhood.pkl")
        with open(f, "wb") as file:
            pickle.dump(neighborhood, file)

                    
    else:
        for i, key in enumerate(self.preprocessor.cell_pos_dict.keys()):
            neighborhood = np.zeros((len(cell_types), len(cell_types)))
            coordinates = self.preprocessor.cell_pos_dict[key]
            # to array
            coordinates = [[np.mean(coordinates[k][0]), np.mean(coordinates[k][1])] for k in sorted(coordinates.keys())]
            assert len(coordinates) == len(self.annotations[i])
            # fit the nearest neighbors
            nbrs = NearestNeighbors(n_neighbors=n_neighbors, algorithm='ball_tree').fit(coordinates)
            for j in range(len(self.annotations[i])):
                indices = nbrs.kneighbors([coordinates[j]], return_distance=False)[0]
                for k in indices[1:]:
                    neighborhood[cell_types.index(self.annotations[i][j]), cell_types.index(self.annotations[i][k])] += 1
            # normalize
            if normalize:
                for ii in range(len(neighborhood)):
                    if neighborhood[ii].sum() > 0:
                        neighborhood[ii] /= neighborhood[ii].sum()
            # plot
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.set_title(f"Neighborhood analysis {i}")
            labels = cell_types
            sns.heatmap(neighborhood, xticklabels=labels, yticklabels=labels, cmap="vlag", linewidth=.5)
            plt.xticks(rotation=60)
            plt.tight_layout()
            f = os.path.join(result_dir, f"{batch_id}_neighborhood_{i}.png")
            plt.savefig(f)
            plt.close()


            # pickle the neighborhood
            f = os.path.join(result_dir, f"{batch_id}_neighborhood_{i}.pkl")
            with open(f, "wb") as file:
                pickle.dump(neighborhood, file)


def _tissue_region_partition(n_clusters=3, f=None):
    # Load the data
    annotation_all = pickle.load(open(f, "rb"))
    x_coords = []
    y_coords = []
    celltypes = []
    cell_id = []

    tissue_labels = []
    for i in range(len(annotation_all)):
        tissue_labels.append({})
        for j in range(len(annotation_all[i])):
            x_coords.append(np.mean(annotation_all[i][j]["Column"]))
            y_coords.append(np.mean(annotation_all[i][j]["Row"]))
            celltypes.append(annotation_all[i][j]["Cell type"])
            cell_id.append(annotation_all[i][j]["Cell ID"])

        # to array
        x_coords = np.array(x_coords)
        y_coords = np.array(y_coords)
        celltypes = np.array(celltypes)

        n_celltypes = np.unique(celltypes)

        n_neighbors = [10, 20, 30, 50, 75, 100, 150, 200]

        # find the nearest 200 neighbors
        nn = NearestNeighbors(n_neighbors=201, algorithm='ball_tree').fit(np.array([x_coords, y_coords]).T)
        distances, indices = nn.kneighbors(np.array([x_coords, y_coords]).T)
        indices = indices[:, 1:]

        compositions = []
        for j in range(len(x_coords)):
            composition = []
            for n in n_neighbors:
                temp = np.zeros(n_celltypes.shape)
                # get index of the neighbor
                idx = indices[j, :n]
                # get the cell type of the neighbor
                cell_types = celltypes[idx]
                # get the count of each cell type
                counts = np.unique(cell_types, return_counts=True)
                # get the proportion of each cell type
                for k in range(len(counts[0])):
                    temp[counts[0][k]] = counts[1][k]
                temp /= np.sum(temp)
                for k in temp:
                    composition.append(k)
            compositions.append(composition)

        compositions = np.array(compositions)
        # HDB clustering
        clusterer = SpectralClustering(n_clusters=n_clusters, affinity='nearest_neighbors', n_neighbors=30)
        cluster_labels = clusterer.fit_predict(compositions)

        for j, id_ in enumerate(cell_id):
            tissue_labels[i][id_] = cluster_labels[j]

    return tissue_labels