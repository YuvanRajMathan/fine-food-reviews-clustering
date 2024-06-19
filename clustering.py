import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from ast import literal_eval
import os

# Load data
datafile_path = "fine_food_reviews_with_embeddings.csv"
df = pd.read_csv(datafile_path)
df["embedding"] = df.embedding.apply(literal_eval).apply(np.array)

# Randomly sample a subset of the dataframe
sampled_df = df.sample(n=500, random_state=42, replace=True)


# Use the sampled dataframe for clustering
matrix = np.vstack(sampled_df.embedding.values)

# Cluster the embeddings
n_clusters = 4
kmeans = KMeans(n_clusters=n_clusters, init="k-means++", random_state=42)
kmeans.fit(matrix)
labels = kmeans.labels_
sampled_df["Cluster"] = labels

# Reduce dimensionality for visualization
tsne = TSNE(n_components=2, perplexity=15, random_state=42, init="random", learning_rate=200)
vis_dims2 = tsne.fit_transform(matrix)

# Visualize the clusters
x = [x for x, y in vis_dims2]
y = [y for x, y in vis_dims2]

for category, color in enumerate(["purple", "green", "red", "blue"]):
    xs = np.array(x)[sampled_df.Cluster == category]
    ys = np.array(y)[sampled_df.Cluster == category]
    plt.scatter(xs, ys, color=color, alpha=0.3)

    avg_x = xs.mean()
    avg_y = ys.mean()

    plt.scatter(avg_x, avg_y, marker="x", color=color, s=100)
plt.title("Clusters identified visualized in language 2d using t-SNE")
plt.show()

