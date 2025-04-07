import pandas as pd
import numpy as np
import scipy.cluster.hierarchy as sch
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import fcluster

# Load the network traffic dataset
data_path = "UNSW_NB15_testing-set.csv"
network_data = pd.read_csv(data_path)

# Select only numerical columns for clustering
# Exclude identifier and label columns
numeric_columns = network_data.select_dtypes(include=['int64', 'float64']).columns.tolist()
numeric_columns.remove('id')       # Remove unique ID column
numeric_columns.remove('label')    # Remove class label column (normal vs. attack)

# Extract only numerical data for clustering
numerical_data = network_data[numeric_columns]

# Standardize the numerical data (mean = 0, std = 1)
standardizer = StandardScaler()
normalized_data = standardizer.fit_transform(numerical_data)

# Perform hierarchical clustering using the 'complete' linkage method
# Limit to first 1000 samples for visualization clarity
plt.figure(figsize=(12, 6))
linkage_data = sch.linkage(normalized_data[:1000], method='complete')

# Draw a dendrogram showing the clustering structure
sch.dendrogram(linkage_data, truncate_mode='level', p=5)
plt.title("Dendrogram - Hierarchical Clustering (Complete Linkage)")
plt.xlabel("Sample Index")
plt.ylabel("Distance")
plt.show()

# Define how many clusters to extract from the dendrogram
target_cluster_count = 5
cluster_labels = fcluster(linkage_data, target_cluster_count, criterion='maxclust')

# Display the first 20 cluster assignments
print("Cluster labels for the first 20 samples:", cluster_labels[:20])

# Use case: Analyzing traffic patterns using clustering
# Add cluster labels to a copy of the first 1000 records
clustered_subset = network_data.iloc[:1000].copy()
clustered_subset['Cluster'] = cluster_labels

# Examine how many normal vs. attack entries are present in each cluster
print("\nNormal vs. attack distribution per cluster:")
print(clustered_subset.groupby(['Cluster', 'label']).size())

# If available, examine attack categories within each cluster
if 'attack_cat' in network_data.columns:
    print("\nAttack category distribution per cluster:")
    print(clustered_subset.groupby(['Cluster', 'attack_cat']).size())
