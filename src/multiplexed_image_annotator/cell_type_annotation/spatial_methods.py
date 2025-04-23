import os
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import SpectralClustering, HDBSCAN

import numpy as np
import pickle


def neighborhood_analysis(annotation_all, n_neighbors=10, cell_types=None, integrate=False, normalize=True, batch_id=None, result_dir=None):
    if integrate:
        neighborhood = np.zeros((len(cell_types), len(cell_types)))
        for i in range(len(annotation_all)):
            x_coords = []
            y_coords = []
            celltypes = []
            cell_id = []
            for j in range(len(annotation_all[i])):
                x_coords.append(np.mean(annotation_all[i][j]["Column"]))
                y_coords.append(np.mean(annotation_all[i][j]["Row"]))
                celltypes.append(annotation_all[i][j]["Cell type"])
                cell_id.append(annotation_all[i][j]["Cell ID"])

            # to array
            x_coords = np.array(x_coords)
            y_coords = np.array(y_coords)
            celltypes = np.array(celltypes).astype(int)
    
            coordinates = np.array([np.array(x_coords), np.array(y_coords)]).T
            assert len(coordinates) == len(annotation_all[i])
            # fit the nearest neighbors
            nbrs = NearestNeighbors(n_neighbors=n_neighbors, algorithm='ball_tree').fit(coordinates)
            for j in range(len(coordinates)):
                indices = nbrs.kneighbors([coordinates[j]], return_distance=False)[0]
                for k in indices[1:]:
                    print()
                    neighborhood[celltypes[j], celltypes[k]] += 1
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

        # write into a csv file
        f = os.path.join(result_dir, f"{batch_id}_integrated_neighborhood.csv")
        with open(f, "w") as file:
            file.write("cell_type,")
            for i in range(len(cell_types)):
                file.write(f"{cell_types[i]},")
            file.write("\n")
            for i in range(len(cell_types)):
                file.write(f"{cell_types[i]},")
                for j in range(len(cell_types)):
                    file.write(f"{neighborhood[i][j]},")
                file.write("\n")

                    
    else:
        for i in range(len(annotation_all)):
            neighborhood = np.zeros((len(cell_types), len(cell_types)))
            x_coords = []
            y_coords = []
            celltypes = []
            cell_id = []
            for j in range(len(annotation_all[i])):
                x_coords.append(np.mean(annotation_all[i][j]["Column"]))
                y_coords.append(np.mean(annotation_all[i][j]["Row"]))
                celltypes.append(annotation_all[i][j]["Cell type"])
                cell_id.append(annotation_all[i][j]["Cell ID"])

            # to array
            x_coords = np.array(x_coords)
            y_coords = np.array(y_coords)
            celltypes = np.array(celltypes).astype(int)

            coordinates = np.array([np.array(x_coords), np.array(y_coords)]).T
            assert len(coordinates) == len(annotation_all[i])
            # fit the nearest neighbors
            nbrs = NearestNeighbors(n_neighbors=n_neighbors, algorithm='ball_tree').fit(coordinates)
            for j in range(len(coordinates)):
                indices = nbrs.kneighbors([coordinates[j]], return_distance=False)[0]
                for k in indices[1:]:
                    neighborhood[cell_types[celltypes[j], celltypes[k]] += 1

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


            # write into a csv file
            f = os.path.join(result_dir, f"{batch_id}_neighborhood_{i}.csv")
            with open(f, "w") as file:
                file.write("cell_type,")
                for i in range(len(cell_types)):
                    file.write(f"{cell_types[i]},")
                file.write("\n")
                for i in range(len(cell_types)):
                    file.write(f"{cell_types[i]},")
                    for j in range(len(cell_types)):
                        file.write(f"{neighborhood[i][j]},")
                    file.write("\n")


def tissue_region_partition(annotation_all, n_clusters=3):

    tissue_labels = []
    for i in range(len(annotation_all)):
        tissue_labels.append({})
        x_coords = []
        y_coords = []
        celltypes = []
        cell_id = []
        for j in range(len(annotation_all[i])):
            x_coords.append(np.mean(annotation_all[i][j]["Column"]))
            y_coords.append(np.mean(annotation_all[i][j]["Row"]))
            celltypes.append(annotation_all[i][j]["Cell type"])
            cell_id.append(annotation_all[i][j]["Cell ID"])

        # to array
        x_coords = np.array(x_coords)
        y_coords = np.array(y_coords)
        celltypes = np.array(celltypes)

        n_celltypes = max(celltypes) + 1

        n_neighbors = [10, 20, 30, 50, 75, 100, 150, 200]

        # find the nearest 200 neighbors
        nn = NearestNeighbors(n_neighbors=201, algorithm='ball_tree').fit(np.array([x_coords, y_coords]).T)
        distances, indices = nn.kneighbors(np.array([x_coords, y_coords]).T)
        indices = indices[:, 1:]

        compositions = []
        for j in range(len(x_coords)):
            composition = []
            for n in n_neighbors:
                temp = np.zeros(n_celltypes)
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

