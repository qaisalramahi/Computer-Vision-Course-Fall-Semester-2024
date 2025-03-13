import numpy as np


def conv_nested(image, kernel):
    """A naive implementation of convolution filter.

    This is a naive implementation of convolution using 4 nested for-loops.
    This function computes convolution of an image with a kernel and outputs
    the result that has the same shape as the input image.

    Args:
        image: numpy array of shape (Hi, Wi).
        kernel: numpy array of shape (Hk, Wk).

    Returns:
        out: numpy array of shape (Hi, Wi).
    """
    Hi, Wi = image.shape[0:2]  # Image dimensions
    Hk, Wk = kernel.shape  # Kernel dimensions

    # Output should have the same size as input
    out = np.zeros((Hi, Wi))

    # Flip the kernel (necessary for convolution)
    kernel_flipped = np.flip(kernel)

    # Calculate padding sizes (since we're NOT adding padding manually)
    pad_height = Hk // 2
    pad_width = Wk // 2

    # Perform convolution using 4 loops
    for i in range(Hi):  
        for j in range(Wi):  
            sum_value = 0  
            for m in range(Hk):  
                for n in range(Wk): 
                    # Subtract the padding size to center the kernel on (i, j)
                    ii = i + m - pad_height
                    jj = j + n - pad_width
                    
                    # Check if (ii, jj) is a valid image coordinate (within bounds)
                    if 0 <= ii < Hi and 0 <= jj < Wi:
                        sum_value += image[ii, jj] * kernel_flipped[m, n]

            # Assign the accumulated sum to the output pixel
            out[i,j] = sum_value

    return out

def zero_pad(image, pad_height, pad_width):
    """ Zero-pad an image.

    Ex: a 1x1 image [[1]] with pad_height = 1, pad_width = 2 becomes:

        [[0, 0, 0, 0, 0],
         [0, 0, 1, 0, 0],
         [0, 0, 0, 0, 0]]         of shape (3, 5)

    Args:
        image: numpy array of shape (H, W).
        pad_width: width of the zero padding (left and right padding).
        pad_height: height of the zero padding (bottom and top padding).

    Returns:
        out: numpy array of shape (H+2*pad_height, W+2*pad_width).
    """

    H, W = image.shape
    out = np.pad(image,((pad_height,pad_height),(pad_width,pad_width)),mode="constant",constant_values=0)

    return out


def conv_fast(image, kernel, flip_kernel = True):
    """ An efficient implementation of convolution filter.

    This function uses element-wise multiplication and np.sum()
    to efficiently compute weighted sum of neighborhood at each
    pixel.

    Hints:
        - Use the zero_pad function you implemented above
        - There should be two nested for-loops
        - You may find np.flip() and np.sum() useful

    Args:
        image: numpy array of shape (Hi, Wi).
        kernel: numpy array of shape (Hk, Wk).

    Returns:
        out: numpy array of shape (Hi, Wi).
    """
    Hi, Wi = image.shape
    Hk, Wk = kernel.shape
    out = np.zeros((Hi, Wi))

    pad_height = Hk // 2
    pad_width = Wk // 2

    if flip_kernel:
        kernel = np.flip(kernel)

    padded_image = zero_pad(image,pad_height,pad_width)

    for i in range(Hi):
        for j in range(Wi):
            area = padded_image[i:i + Hk, j : j + Wk ]
            out[i,j] = np.sum(area * kernel)

    return out

def conv_faster(image, kernel):
    """
    Args:
        image: numpy array of shape (Hi, Wi).
        kernel: numpy array of shape (Hk, Wk).

    Returns:
        out: numpy array of shape (Hi, Wi).
    """
    Hi, Wi = image.shape
    Hk, Wk = kernel.shape
    out = np.zeros((Hi, Wi))

    ### YOUR CODE HERE
    pass
    ### END YOUR CODE

    return out

def cross_correlation(f, g):
    """ Cross-correlation of f and g.

    Hint: use the conv_fast function defined above.

    Args:
        f: numpy array of shape (Hf, Wf).
        g: numpy array of shape (Hg, Wg).

    Returns:
        out: numpy array of shape (Hf, Wf).
    """

    out = conv_fast(f,g,flip_kernel=False)

    return out

def zero_mean_cross_correlation(f, g):
    """ Zero-mean cross-correlation of f and g.

    Subtract the mean of g from g so that its mean becomes zero.

    Hint: you should look up useful numpy functions online for calculating the mean.

    Args:
        f: numpy array of shape (Hf, Wf).
        g: numpy array of shape (Hg, Wg).

    Returns:
        out: numpy array of shape (Hf, Wf).
    """

    g_mean = g - np.mean(g)

    out = conv_fast(f, g_mean, flip_kernel=False)
    return out


    """ Normalized cross-correlation of f and g.

    Normalize the subimage of f and the template g at each step
    before computing the weighted sum of the two.

    Hint: you should look up useful numpy functions online for calculating 
          the mean and standard deviation.

    Args:
        f: numpy array of shape (Hf, Wf).
        g: numpy array of shape (Hg, Wg).

    Returns:
        out: numpy array of shape (Hf, Wf).
    """

    """ Hf, Wf = f.shape
    Hg, Wg = g.shape  

    out = np.zeros((Hf, Wf)) 

    g_mean = np.mean(g)  
    g_std = np.std(g)

    

    for i in range(Hf - Hg + 1):
        for j in range(Wf - Wg + 1):

            patch_img = f[i:i + Hg, j:j + Wg]

            
            patch_mean = np.mean(patch_img)
            patch_std = np.std(patch_img) 
            

            # Ensure we don't divide by zero for very flat regions
            if patch_std != 0 and g_std != 0:
                # Normalize the patch and template
                normalized_patch = (patch_img - patch_mean) / patch_std
                normalized_template = (g - g_mean) / g_std

                # Compute the normalized cross-correlation
                out[i, j] = np.sum(normalized_patch * normalized_template)

    return out """
def normalized_cross_correlation(f, g):
    """ 
    Normalized cross-correlation of f and g.

    Normalize the subimage of f and the template g at each step
    before computing the weighted sum of the two.

    Args:
        f: numpy array of shape (Hf, Wf).
        g: numpy array of shape (Hg, Wg).

    Returns:
        out: numpy array of shape (Hf, Wf).
    """

    Hf, Wf = f.shape
    Hg, Wg = g.shape  

    out = np.zeros((Hf, Wf)) 

    g_mean = np.mean(g)  
    g_std = np.std(g)

    # We can look at this as the 'hypothetical' padding height and width
    change_y = Hg // 2
    change_x = Wg // 2

    for m in range(change_y, Hf - change_y):
        for n in range(change_x, Wf - change_x):
            f_patch = f[m - change_y : m + change_y , n - change_x: n + change_x + 1]
            patch_mean = np.mean(f_patch)
            patch_std = np.std(f_patch)

            if patch_std != 0 and g_std != 0:
                f_norm = (f_patch - patch_mean) / patch_std
                g_norm = (g - g_mean) / g_std
                out[m, n] = np.sum(f_norm * g_norm)

    return out

