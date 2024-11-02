import torch
import torch.nn.functional as F
from torch.nn import PairwiseDistance

def cluster_code_assignment(input_tensor, cluster_prototypes, p=2):
    """
    Compute the cluster codes using Wasserstein distance.

    Args:
        input_tensor (torch.Tensor): Input tensor of shape (N, D), where N is the number of data points and D is the dimensionality.
        cluster_prototypes (torch.Tensor): Cluster prototypes of shape (K, D), where K is the number of clusters.
        p (int): The order of the Wasserstein distance.

    Returns:
        cluster_codes (torch.Tensor): Cluster codes of shape (N,), representing the assigned cluster for each data point.
    """
    N, D = input_tensor.shape
    K = cluster_prototypes.shape[0]

    pairwise_distance = PairwiseDistance(p=p)

    cluster_codes = torch.zeros(N, D, dtype=torch.long, device=input_tensor.device)

    for i in range(N):
        min_distance = float('inf')
        min_cluster = -1

        for j in range(K):
            distance = torch.pow(pairwise_distance(input_tensor[i].unsqueeze(0), cluster_prototypes[j].unsqueeze(0)), p)
            if distance < min_distance:
                min_distance = distance
                min_cluster = j

        #print("Cluster")
        #print(min_cluster)
        cluster_codes[i] = min_cluster

    return cluster_codes



# Generate random tensor data
num_samples = 100
num_features = 10
num_clusters = 5
num_iterations = 100

# Random input tensor of shape (num_samples, num_features) on CUDA
input_tensor = torch.randn(num_samples, num_features).cuda()

# Random cluster code tensor of shape (num_samples, num_clusters) on CUDA
cluster_code = torch.randn(num_samples, num_clusters).cuda()

# Normalize the cluster code tensor to sum to 1 along the cluster dimension
cluster_code = torch.softmax(cluster_code, dim=1)

# Iterate IPF
for _ in range(num_iterations):
    # Compute row sums and column sums of the cluster code tensor
    row_sums = cluster_code.sum(dim=1, keepdim=True)
    col_sums = cluster_code.sum(dim=0, keepdim=True)

    # Normalize the cluster code tensor by row and column sums
    cluster_code = cluster_code / row_sums
    cluster_code = cluster_code / col_sums

# Compute the cluster assignment by selecting the maximum value along the cluster dimension
_, cluster_assignment = cluster_code.max(dim=1)

# Print cluster assignment
print("Cluster Assignment:")
print(cluster_assignment)
