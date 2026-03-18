import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

np.random.seed(0)

# Normal customers
spend1 = np.random.randint(100,300,(20,1))
visit1 = np.random.randint(1,4,(20,1))
normal = np.hstack((spend1, visit1))

# Heavy customers
spend2 = np.random.randint(600,900,(20,1))
visit2 = np.random.randint(5,9,(20,1))
heavy = np.hstack((spend2, visit2))

# Combine data
data = np.vstack((normal, heavy))

# K-Means
model = KMeans(n_clusters=2)
model.fit(data)

labels = model.labels_
centers = model.cluster_centers_

# Plot
plt.scatter(data[:,0], data[:,1], c=labels)
plt.scatter(centers[:,0], centers[:,1], marker='X', s=200)

plt.xlabel("Money Spent")
plt.ylabel("Visits")
plt.title("Customer Clusters")

plt.show()