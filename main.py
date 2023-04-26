import numpy as np
import matplotlib.pyplot as plt

# datasets
from sklearn import datasets

# pyclustering kmeans
from pyclustering.cluster.kmeans import kmeans
from pyclustering.cluster.kmedians import kmedians
from pyclustering.utils.metric import distance_metric
from pyclustering.cluster.center_initializer import random_center_initializer, kmeans_plusplus_initializer

# streamlit
import streamlit as st

np.random.seed(0)

# Create a streamlit app
st.title("Clustering using Different Metrics")
st.write(
    "Visualizing the impact of different distance metrics on clustering results, showing how Euclidean, Manhattan, and other metrics lead to different cluster formations. Have a dropdown for distance metrics."
)

st.write("Let x and y be two n dimensional vectors.")
st.write("1. Euclidean Distance Formula:")
st.latex(r'''
d = \sqrt{\sum_{i = 1}^{i = n} (x_i - y_i)^2}
''')

st.write("2. Squared Euclidean Distance Formula:")
st.latex(r'''
d = \sum_{i = 1}^{i = n} (x_i - y_i)^2
''')

st.write("3. Manhattan Distance Formula:")
st.latex(r'''
d = \sum_{i = 1}^{i = n} \mid x_i - y_i \mid
''')

st.write("4. Chebyshev Distance Formula:")
st.latex(r'''
d = max_{i = 1}^{i = n} \mid x_i - y_i \mid
''')

st.write("5. Canberra Distance Formula:")

st.latex(r'''
d = \sum_{i = 1}^{i = n} \frac{\mid x_i - y_i \mid}{\mid x_i \mid + \mid y_i \mid}
''')

st.write("6. Chi-Squared Distance Formula:")

st.latex(r'''
d = \frac{1}{2} \sum_{i = 1}^{i = n} \frac{(x_i - y_i)^2}{(x_i + y_i)}
''')

# The sidebar contains the sliders
with st.sidebar:
    # Drop down to select Clustering Algorithm
    clustering_algorithm = st.selectbox(
    'Clustering Algorithm',
    ('K-Means', 'K-Medians'))

    # Drop down to select Dataset
    clustering_dataset = st.selectbox(
    'Dataset',
    ('Noisy Circles', 'Noisy Moons', 'Blobs', 'Anisotropicly Distributed', 'Varied Variances', 'No Structure'))

    # Create a slider for n_samples
    n_samples = st.slider("Number of Data Points", 1, 100000, 1500)

    # Create a slider for n_clusters
    n_clusters = st.slider("Number of Clusters", 1, 20, 5)

    # Drop down to select Distance Metric
    distance_metric_to_use = st.selectbox(
    'Distance Metric',
    ('Euclidean', 'Squared Euclidean', 'Manhattan', 'Chebyshev', 'Canberra', 'Chi-Square'))

    # Drop down to select Center Initialization Method for K-Means
    center_initialization = st.selectbox(
    'Center Initialization',
    ('Random Initialization', 'K-Means++ Initialization'))

# ============
# Generate datasets. We choose the size big enough to see the scalability
# of the algorithms, but not too big to avoid too long running times
# ============
noisy_circles = datasets.make_circles(n_samples=n_samples, factor=0.5, noise=0.05)
noisy_moons = datasets.make_moons(n_samples=n_samples, noise=0.05)
blobs = datasets.make_blobs(n_samples=n_samples, random_state=8)
no_structure = np.random.rand(n_samples, 2), None

# Anisotropicly distributed data
random_state = 170
X, y = datasets.make_blobs(n_samples=n_samples, random_state=random_state)
transformation = [[0.6, -0.6], [-0.4, 0.8]]
X_aniso = np.dot(X, transformation)
aniso = (X_aniso, y)

# blobs with varied variances
varied = datasets.make_blobs(
    n_samples=n_samples, cluster_std=[1.0, 2.5, 0.5], random_state=random_state
)

clustering_datasets = {
    "Noisy Circles": noisy_circles,
    "Noisy Moons": noisy_moons,
    "Blobs": blobs,
    "Anisotropicly Distributed": aniso,
    "Varied Variances": varied,
    "No Structure": no_structure
}

distance_metrics = {
    'Euclidean': 0, 
    'Squared Euclidean': 1, 
    'Manhattan': 2, 
    'Chebyshev': 3, 
    'Canberra': 5, 
    'Chi-Square': 6
}

# Clustering
X, y = clustering_datasets[clustering_dataset]

if (center_initialization == "K-Means++ Initialization"):
    initial_centers = kmeans_plusplus_initializer(X, n_clusters, random_state=5).initialize()
else:
    initial_centers = random_center_initializer(X, n_clusters, random_state=5).initialize()

# instance created for respective distance metric
if (clustering_algorithm == "K-Means"):
    instance = kmeans(X, initial_centers=initial_centers, metric=distance_metric(distance_metrics[distance_metric_to_use]))
elif (clustering_algorithm == "K-Medians"):
    instance = kmedians(X, initial_medians=initial_centers, metric=distance_metric(distance_metrics[distance_metric_to_use]))

# perform cluster analysis
instance.process()
# cluster analysis results - clusters and centers
pyClusters = instance.get_clusters()
if (clustering_algorithm == "K-Means"):
    pyCenters = instance.get_centers()
elif (clustering_algorithm == "K-Medians"):
    pyCenters = instance.get_medians()

# Plot the data
colors = np.array(["red","green","blue","yellow","pink","black","orange","purple","beige","brown","gray","cyan","magenta"])
fig, ax = plt.subplots()

for j in range(n_clusters):
    # # Region
    # drawObject = plt.Circle(
    #     (pyCenters[j][0], pyCenters[j][1]), 
    #     radius=0.1, 
    #     fill=False, 
    #     color="black"
    # )
    # ax.add_artist(drawObject)

    # Data
    ax.scatter(
        [X[i][0] for i in pyClusters[j]], [X[i][1] for i in pyClusters[j]],
        s=10, c=colors[j % len(colors)],
        marker='o', edgecolor='black',
        label='Cluster ' + str(j)
    )

    # Centroids
    ax.scatter(
        [pyCenters[j][0]], [pyCenters[j][1]],
        s=200, marker='*',
        c=colors[j % len(colors)], edgecolor='black',
        label='Centroid ' + str(j)
    )

ax.set_xlabel("x")
ax.set_ylabel("y")
fig.legend()
st.pyplot(fig)