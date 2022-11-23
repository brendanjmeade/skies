import datetime
import numpy as np
import scipy

# Matrix size and number of eigenvalues for truncated case
n_pts = 1000
n_eigenvalues = 100

# Set of coordinates for synthetic distance calculation
x = np.random.rand(n_pts)
y = np.random.rand(n_pts)
z = np.random.rand(n_pts)
centroid_coordinates = np.array([x, y, z]).T
distance_matrix = scipy.spatial.distance.cdist(
    centroid_coordinates, centroid_coordinates, "euclidean"
)
distance_matrix = (distance_matrix - np.min(distance_matrix)) / np.ptp(distance_matrix)
correlation_matrix = np.exp(-distance_matrix)

# Full vs. truncated eigenvalue experiments
start_time = datetime.datetime.now()
eigenvalues_truncated, eigenvectors_truncated = scipy.linalg.eigh(
    correlation_matrix,
    subset_by_index=[n_pts - n_eigenvalues, n_pts - 1],
)
end_time = datetime.datetime.now()
print(f"Truncated EV time: {(end_time - start_time)}")

start_time = datetime.datetime.now()
eigenvalues, eigenvectors = scipy.linalg.eigh(correlation_matrix)
end_time = datetime.datetime.now()
print(f"Full EV time     : {(end_time - start_time)}")

print(np.allclose(eigenvalues[-n_eigenvalues:], eigenvalues_truncated))
