import torch
import numpy as np
from sklearn.metrics import pairwise_distances_chunked
from sklearn.neighbors import KDTree

def kmeans_gpu(data, k, num_iters=100):
    # Initialize centroids randomly
    centroids = data[np.random.choice(data.shape[0], k, replace=False)]
    
    data_cuda = data
    centroids_cuda = centroids
    
    # Main K-means iteration
    for _ in range(num_iters):
        # Calculate distances between data points and centroids
        distances = torch.sum((data_cuda[:, None] - centroids_cuda[None]) ** 2, dim=-1)

        # Assign data points to the closest centroid
        labels = torch.argmin(distances, dim=-1)

        # Update centroids based on assigned data points
        for i in range(k):
            if torch.sum(labels == i) > 0:
                centroids_cuda[i] = torch.mean(data_cuda[labels == i], dim=0)

    return centroids_cuda.cpu()

def dbscan_gpu(data, eps, min_samples, num_iters=100, lr=0.1):
    # Convert data to CUDA tensor
    data_cuda = data
    #data_cuda = torch.from_numpy(data).float().cuda()

    # Initialize random cluster labels
    labels = torch.randint(low=-1, high=0, size=(data.shape[0],)).cuda()

    # Enable gradients for data tensor
    #data_cuda.requires_grad = True

    # Main DBSCAN iteration
    for _ in range(num_iters):
        # Calculate pairwise distances between data points
        distances = torch.cdist(data_cuda, data_cuda)

        # Find neighbors within epsilon distance
        neighbors = (distances <= eps).float()

        # Count the number of neighbors for each point
        num_neighbors = torch.sum(neighbors, dim=1)

        # Expand clusters and update labels
        for i in range(data.shape[0]):
            if labels[i] != -1 or num_neighbors[i] < min_samples:
                continue

            # Compute distances to all neighbors
            distances_to_neighbors = distances[i] * neighbors[i]

            # Compute gradients of distances with respect to data points
            gradients = torch.autograd.grad(torch.sum(distances_to_neighbors), data_cuda, retain_graph=True)[0]

            # Update data points using gradients
            data_cuda.data[i] -= lr * gradients[i]

        # Assign cluster labels based on updated data points
        for i in range(data.shape[0]):
            if labels[i] == -1 and num_neighbors[i] >= min_samples:
                labels[i] = 1

    # Transfer labels back to CPU and return
    labels = labels.cpu()
    return labels


def affinity_propagation_gpu(data, damping=0.5, max_iter=200, convergence_iter=15):
    # Convert data to CUDA tensor
    data_cuda = data
    #data_cuda = torch.from_numpy(data).float().cuda()

    num_samples = data.shape[0]

    # Initialize similarity matrix and preference on GPU
    similarity_matrix = torch.zeros(num_samples, num_samples).float().cuda()
    preference = torch.zeros(num_samples, 1).float().cuda()

    # Enable gradients for similarity matrix and preference
    similarity_matrix.requires_grad = True
    preference.requires_grad = True

    # Main Affinity Propagation iteration
    for _ in range(max_iter):
        # Calculate responsibility matrix
        max_similarity, _ = torch.max(similarity_matrix, dim=1, keepdim=True)
        responsibility = similarity_matrix - max_similarity
        responsibility = responsibility + preference
        responsibility = torch.clamp(responsibility, min=0)

        # Calculate availability matrix
        availability = torch.zeros_like(responsibility)
        availability_t = availability.t()
        responsibility_t = responsibility.t()
        availability[torch.arange(num_samples), torch.argmax(responsibility_t, dim=1)] = 1
        availability_t[torch.arange(num_samples), torch.argmax(responsibility, dim=1)] = 1
        availability[torch.arange(num_samples), torch.arange(num_samples)] += torch.sum(responsibility_t > 0, dim=1)
        availability_t[torch.arange(num_samples), torch.arange(num_samples)] += torch.sum(responsibility > 0, dim=1)
        availability = torch.min(availability, availability_t)

        # Update responsibility and availability matrices using damping factor
        responsibility = damping * responsibility + (1 - damping) * responsibility
        availability = damping * availability + (1 - damping) * availability

    # Transfer responsibility and availability matrices back to CPU
    responsibility_cpu = responsibility.detach().cpu().numpy()
    availability_cpu = availability.detach().cpu().numpy()

    # Perform final clustering on CPU
    labels = np.argmax(responsibility_cpu + availability_cpu, axis=1)
    res = torch.from_numpy(labels)
    return res


