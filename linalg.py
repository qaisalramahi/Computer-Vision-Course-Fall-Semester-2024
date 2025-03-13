import numpy as np
import math


def dot_product(a, b):
    """Implement dot product between the two vectors: a and b.

    (optional): While you can solve this using for loops, we recommend
    that you look up `np.dot()` online and use that instead.

    Args:
        a: numpy array of shape (x, n)
        b: numpy array of shape (n, x)

    Returns:
        out: numpy array of shape (x, x) (scalar if x = 1)
    """
    out = np.dot(a,b) 
    
    ### END YOUR CODE
    return out


def complicated_matrix_function(M, a, b):
    """Implement (a * b) * (M * a.T).

    (optional): Use the `dot_product(a, b)` function you wrote above
    as a helper function.

    Args:
        M: numpy matrix of shape (x, n).
        a: numpy array of shape (1, n).
        b: numpy array of shape (n, 1).

    Returns:
        out: numpy matrix of shape (x, 1).
    """
    scalar_component = dot_product(a,b)
    matrix_multiplication = dot_product(M,a.T)
    
    
    out = scalar_component * matrix_multiplication
    if out.ndim == 1:
        out = out[:,np.newaxis] # convert 1d to 2d

    return out


def eigen_decomp(M):
    """Implement eigenvalue decomposition.

    (optional): You might find the `np.linalg.eig` function useful.

    Args:
        matrix: numpy matrix of shape (m, m)

    Returns:
        w: numpy array of shape (m,) such that the column v[:,i] is the eigenvector corresponding to the eigenvalue w[i].
        v: Matrix where every column is an eigenvector.
    """
    eigenvalues, eigenvectors = np.linalg.eig(M)
    w = eigenvalues
    v = eigenvectors
    
    return w, v


def euclidean_distance_native(u, v):
    """Computes the Euclidean distance between two vectors, represented as Python
    lists.

    Args:
        u (List[float]): A vector, represented as a list of floats.
        v (List[float]): A vector, represented as a list of floats.

    Returns:
        float: Euclidean distance between `u` and `v`.
    """
    # First, run some checks:
    assert isinstance(u, list)
    assert isinstance(v, list)
    assert len(u) == len(v)

    # Compute the distance!
    # Notes:
    #  1) Try breaking this problem down: first, we want to get
    #     the difference between corresponding elements in our
    #     input arrays. Then, we want to square these differences.
    #     Finally, we want to sum the squares and square root the
    #     sum.

    sum_of_diffsquared = 0
    for i in range(len(u)):
       diff = u[i]-v[i]
       sum_of_diffsquared +=  diff ** 2
        

    out = math.sqrt(sum_of_diffsquared)
    
    return out

    ### END YOUR CODE


def euclidean_distance_numpy(u, v):
    """Computes the Euclidean distance between two vectors, represented as NumPy
    arrays.

    Args:
        u (np.ndarray): A vector, represented as a NumPy array.
        v (np.ndarray): A vector, represented as a NumPy array.

    Returns:
        float: Euclidean distance between `u` and `v`.
    """
    # First, run some checks:
    assert isinstance(u, np.ndarray)
    assert isinstance(v, np.ndarray)
    assert u.shape == v.shape

    # Compute the distance!
    # Note:
    #  1) You shouldn't need any loops
    #  2) Some functions you can Google that might be useful:
    #         np.sqrt(), np.sum()
    #  3) Try breaking this problem down: first, we want to get
    #     the difference between corresponding elements in our
    #     input arrays. Then, we want to square these differences.
    #     Finally, we want to sum the squares and square root the
    #     sum.
    diff = u - v

    squares = diff ** 2

    sum_of_squares = np.sum(squares)

    out = np.sqrt(sum_of_squares)
    
    return out

    ### END YOUR CODE


def get_eigen_values_and_vectors(M, k):

    """Return top k eigenvalues and eigenvectors of matrix M. By top k
    here we mean the eigenvalues with the top ABSOLUTE values (lookup
    np.argsort for a hint on how to do so.)

    (optional): Use the `eigen_decomp(M)` function you wrote above
    as a helper function

    Args:
        M: numpy matrix of shape (m, m).
        k: number of eigen values and respective vectors to return.

    Returns:
        eigenvalues: list of length k containing the top k eigenvalues
        eigenvectors: list of length k containing the top k eigenvectors
            of shape (m,)
    """

    top_eigenvalues = []
    top_eigenvectors = []

    eigenvalues, eigenvectors = eigen_decomp(M)
    position_of_eigen = np.argsort(np.abs(eigenvalues)) [::-1] # for decreasing order because we want the top values first

    for i in range(k):
         idx = position_of_eigen[i]
         top_eigenvalues.append(eigenvalues[idx])
         top_eigenvectors.append(eigenvectors[:, idx]) # means any row at the specific eigenth column
        
    return np.array(top_eigenvalues) , np.array(top_eigenvectors)

'''M = np.array([[4, -1, 2],  # Shape (3, 3)
              [3, -2, 1],
              [1, 4, 3]])

M1 = np.array([[4, -2],
              [1,  1]])

a = np.array([[1, 2, 3]])  # Shape (1, 3) 
b = np.array([[1],         # Shape (3, 1) 
              [5],
              [8]])
k = 2
u = np.array([2,5,6])
v = np.array([7,8,9])
u_list = u.tolist() # for normal method which takes in a list not an np array
v_list = v.tolist()
print(u.shape)
print(v.shape)
print(f"eigen decompostion: {eigen_decomp(M1)}")
eigenvalues, eigenvectors = get_eigen_values_and_vectors(M, k)
print(f"Top Eigenvalues: {eigenvalues}")
print(f"Top Eigenvectors: {eigenvectors}")
print(f"euclidian distance using normal methods: {euclidean_distance_native(u_list,v_list)}")
print(f"euclidian distance using numpy: {euclidean_distance_numpy(u,v)}")

print(complicated_matrix_function(M,a,b))
print()'''