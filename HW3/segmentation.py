import numpy as np
import random
from scipy.spatial.distance import squareform, pdist
from skimage.util import img_as_float

### Clustering Methods
def kmeans(features, k, num_iters=100):
    """ Use kmeans algorithm to group features into k clusters.

    K-Means algorithm can be broken down into following steps:
        1. Randomly initialize cluster centers
        2. Assign each point to the closest center
        3. Compute new center of each cluster
        4. Stop if cluster assignments did not change
        5. Go to step 2

    Args:
        features - Array of N features vectors. Each row represents a feature
            vector.
        k - Number of clusters to form.
        num_iters - Maximum number of iterations the algorithm will run.

    Returns:
        assignments - Array representing cluster assignment of each point.
            (e.g. i-th point is assigned to cluster assignments[i])
    """

    N, D = features.shape

    assert N >= k, 'Number of clusters cannot be greater than number of points'

    # Randomly initalize cluster centers
    idxs = np.random.choice(N, size=k, replace=False)
    centers = features[idxs]
    assignments = np.zeros(N)

    for n in range(num_iters):
        # Step 2: Assign each point to the closest center
        new_assignments = np.array([np.argmin([np.linalg.norm(feature - center) for center in centers]) for feature in features])

        # Step 3: Update centers based on current assignments
        for i in range(k): 
            centers[i] = features[new_assignments==i].mean(axis=0) # all the feature points that belong to a center and hence cluster

        # Step 4: Check for convergence
        if np.array_equal(assignments, new_assignments):
            break
        
        # Update assignments
        assignments = new_assignments

    return assignments

def kmeans_fast(features, k, num_iters=100):
    """ Use kmeans algorithm to group features into k clusters.

    This function makes use of numpy functions and broadcasting to speed up the
    first part(cluster assignment) of kmeans algorithm.

    Hints
    - You may find np.repeat and np.argmin useful

    Args:
        features - Array of N features vectors. Each row represents a feature
            vector.
        k - Number of clusters to form.
        num_iters - Maximum number of iterations the algorithm will run.

    Returns:
        assignments - Array representing cluster assignment of each point.
            (e.g. i-th point is assigned to cluster assignments[i])
    """

    N, D = features.shape

    assert N >= k, 'Number of clusters cannot be greater than number of points'

    # Randomly initalize cluster centers
    idxs = np.random.choice(N, size=k, replace=False)
    centers = features[idxs]
    assignments = np.zeros(N,dtype=int)

    for n in range(num_iters):

        '''I wrote many comments because I myself was trying my best to understand everything :)'''
        # we want to calculate the distance between every point and cluster
        # so we need to have the points repeated to each center (k times)
        # and we need to have each center cycled through N times
        repeated_features = np.repeat(features,k,axis=0)  # N,D originally in shape with repeat we make it N*k, D
        repeated_centers = np.tile(centers,(N,1)) # k,D orignally so multiply and get N*k,D- same shape array as repeated_features

        distances = np.linalg.norm(repeated_features - repeated_centers, axis=1).reshape(N,k)
        new_assignments = np.argmin(distances, axis = 1)

               # Step 3: Update centers based on current assignments
        for i in range(k): 
            if np.any(new_assignments==i): # in case any are 0 not to divide by 0
                centers[i] = features[new_assignments==i].mean(axis=0) # all the feature points that belong to a center and hence cluster

        # Step 4: Check for convergence
        if np.array_equal(assignments, new_assignments):
            break
        
        # Update assignments
        assignments = new_assignments

    return assignments

def hierarchical_clustering(features, k):
    """ Run the hierarchical agglomerative clustering algorithm.

    The algorithm is conceptually simple:

    Assign each point to its own cluster
    While the number of clusters is greater than k:
        Compute the distance between all pairs of clusters
        Merge the pair of clusters that are closest to each other

    We will use Euclidean distance to define distance between clusters.

    Recomputing the centroids of all clusters and the distances between all
    pairs of centroids at each step of the loop would be very slow. Thankfully
    most of the distances and centroids remain the same in successive
    iterations of the outer loop; therefore we can speed up the computation by
    only recomputing the centroid and distances for the new merged cluster.

    Even with this trick, this algorithm will consume a lot of memory and run
    very slowly when clustering large set of points. In practice, you probably
    do not want to use this algorithm to cluster more than 10,000 points.

    Args:
        features - Array of N features vectors. Each row represents a feature
            vector.
        k - Number of clusters to form.

    Returns:
        assignments - Array representing cluster assignment of each point.
            (e.g. i-th point is assigned to cluster assignments[i])
    """
    N, D = features.shape

    assert N >= k, 'Number of clusters cannot be greater than number of points'

    # Assign each point to its own cluster
    assignments = np.arange(N)
    centers = np.copy(features)
    n_clusters = N
    # compute the distances between all pairs of clusters 
    while n_clusters > k:
        d_min = np.linalg.norm(centers[0] - centers[1])
        dist1, dist2 = 0, 1       # original distance indices
        for i in range(n_clusters - 1):
            for j in range(i + 1, n_clusters):
                distance = np.linalg.norm(centers[i] - centers[j])
                if distance < d_min:
                    d_min, dist1, dist2 = distance, i, j

        # merge clusters
        # centoid of the merged clusters assigned to 1st center and is = to average link
        centers[dist1] = (centers[dist1] + centers[dist2]) / 2
        centers = np.delete(centers, dist2, 0)

        for i in range(N):
            if (assignments[i] > dist2 - 1):
                if (assignments[i] == dist2):
                    assignments[i] = dist1
                else:
                    assignments[i] -= 1

        n_clusters -= 1

    return assignments



### Pixel-Level Features
def color_features(img):
    """ Represents a pixel by its color.

    Args:
        img - array of shape (H, W, C)

    Returns:
        features - array of (H * W, C)
    """
    H, W, C = img.shape
    img = img_as_float(img)
    features = np.zeros((H*W, C))

    features = img.reshape(H*W,C)
    ### END YOUR CODE

    return features

def color_position_features(img):
    """ Represents a pixel by its color and position.

    Combine pixel's RGB value and xy coordinates into a feature vector.
    i.e. for a pixel of color (r, g, b) located at position (x, y) in the
    image. its feature vector would be (r, g, b, x, y).

    Don't forget to normalize features.

    Hints
    - You may find np.mgrid and np.dstack useful
    - You may use np.mean and np.std

    Args:
        img - array of shape (H, W, C)

    Returns:
        features - array of (H * W, C+2)
    """
    H, W, C = img.shape
    color = img_as_float(img)
    features = np.zeros((H*W, C+2))

    x,y = np.mgrid[0:H,0:W]
    pos = np.dstack((x,y)) # shape of H, W ,2 
    pos = (pos - np.min(pos))/(np.max(pos) - np.min(pos)) # normalization
    temp = np.dstack((color,pos)) # Shape of H,W, C+2 
    features = temp.reshape((H*W, C+2)) # 3d to 2d
    return features

def my_features(img):
    """ Implement your own features

    Args:
        img - array of shape (H, W, C)

    Returns:
        features - array of (H * W, C)
    """
    features = None
    ### YOUR CODE HERE
    pass
    ### END YOUR CODE
    return features


### Quantitative Evaluation
def compute_accuracy(mask_gt, mask):
    """ Compute the pixel-wise accuracy of a foreground-background segmentation
        given a ground truth segmentation.

    Args:
        mask_gt - The ground truth foreground-background segmentation. A
            logical of size H x W where mask_gt[y, x] is 1 if and only if
            pixel (y, x) of the original image was part of the foreground.
        mask - The estimated foreground-background segmentation. A logical
            array of the same size and format as mask_gt.

    Returns:
        accuracy - The fraction of pixels where mask_gt and mask agree. A
            bigger number is better, where 1.0 indicates a perfect segmentation.
    """
    H,W = mask.shape
    count = 0
    for i in range(H):
        for j in range(W):
            if mask[i,j] == mask_gt[i,j]:
                count += 1

    accuracy = count / (H*W)

    return accuracy

def evaluate_segmentation(mask_gt, segments):
    """ Compare the estimated segmentation with the ground truth.

    Note that 'mask_gt' is a binary mask, while 'segments' contain k segments.
    This function compares each segment in 'segments' with the ground truth and
    outputs the accuracy of the best segment.

    Args:
        mask_gt - The ground truth foreground-background segmentation. A
            logical of size H x W where mask_gt[y, x] is 1 if and only if
            pixel (y, x) of the original image was part of the foreground.
        segments - An array of the same size as mask_gt. The value of a pixel
            indicates the segment it belongs.

    Returns:
        best_accuracy - Accuracy of the best performing segment.
            0 <= accuracy <= 1, where 1.0 indicates a perfect segmentation.
    """

    num_segments = np.max(segments) + 1
    best_accuracy = 0

    # Compare each segment in 'segments' with the ground truth
    for i in range(num_segments):
        mask = (segments == i).astype(int)
        accuracy = compute_accuracy(mask_gt, mask)
        best_accuracy = max(accuracy, best_accuracy)

    return best_accuracy
