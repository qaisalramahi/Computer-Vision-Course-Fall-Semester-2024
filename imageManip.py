import math

import numpy as np
from PIL import Image
from skimage import color, io


def load(image_path):
    """Loads an image from a file path.

    HINT: Look up `skimage.io.imread()` function.

    Args:
        image_path: file path to the image.

    Returns:
        out: numpy array of shape(image_height, image_width, 3).
    """
    out = io.imread(image_path)
    
    
    # Let's convert the image to be between the correct range.
    out = out.astype(np.float64) / 255
    return out


def crop_image(image, start_row, start_col, num_rows, num_cols):
    """Crop an image based on the specified bounds.

    Args:
        image: numpy array of shape(image_height, image_width, 3).
        start_row (int): The starting row index we want to include in our cropped image.
        start_col (int): The starting column index we want to include in our cropped image.
        num_rows (int): Number of rows in our desired cropped image.
        num_cols (int): Number of columns in our desired cropped image.

    Returns:
        out: numpy array of shape(num_rows, num_cols, 3).
    """
    # python slicing 
    out = image[start_row : start_row + num_rows , start_col : start_col + num_cols]

    
    ### END YOUR CODE

    return out


def dim_image(image):
    """Change the value of every pixel by following

                        x_n = 0.5*x_p^2

    where x_n is the new value and x_p is the original value.

    Args:
        image: numpy array of shape(image_height, image_width, 3).

    Returns:
        out: numpy array of shape(image_height, image_width, 3).
    """
    x_p = image
    x_n = 0.5 * (x_p ** 2)

    out = x_n


    return out


def resize_image(input_image, output_rows, output_cols):
    """Resize an image using the nearest neighbor method.

    Args:
        input_image (np.ndarray): RGB image stored as an array, with shape
            `(input_rows, input_cols, 3)`.
        output_rows (int): Number of rows in our desired output image.
        output_cols (int): Number of columns in our desired output image.

    Returns:
        np.ndarray: Resized image, with shape `(output_rows, output_cols, 3)`.
    """
    input_rows, input_cols, channels = input_image.shape
    assert channels == 3

    # 1. Create the resized output image
    output_image = np.zeros(shape=(output_rows, output_cols, 3))
    
    row_scale_factor = input_rows / output_rows # let it be = to 10/2 = 5
    col_scale_factor = input_cols / output_cols # every 5 rows/cols in the input image = output
    
    # input: (0,1) -> output: (0,5) therefore shrinking

    # 2. Populate the `output_image` array using values from `input_image`
    #    > This should require two nested for loops!
    for i in range(output_rows):
        for j in range(output_cols):
            # 3. Determine the corresponding coordinates in the input image
            input_i = int(i * row_scale_factor)
            input_j = int(j * col_scale_factor)
            # 4. Copy the value from the input image to the output image
            output_image[i,j, :] = input_image[input_i,input_j, :]


    # 3. Return the output image
    return output_image


def rotate2d(point, theta):
    """Rotate a 2D coordinate by some angle theta.

    Args:
        point (np.ndarray): A 1D NumPy array containing two values: an x and y coordinate.
        theta (float): An theta to rotate by, in radians.

    Returns:
        np.ndarray: A 1D NumPy array containing your rotated x and y values.
    """
    assert point.shape == (2,)
    assert isinstance(theta, float)

    # Reminder: np.cos() and np.sin() will be useful here!
    x = point[0]
    y = point[1]
    new_x = x * np.cos(theta) - y * np.sin(theta)
    new_y = x * np.sin(theta) + y * np.cos(theta)

    return np.array([new_x, new_y])


def rotate_image(input_image, theta):
    """Rotate an image by some angle theta.

    Args:
        input_image (np.ndarray): RGB image stored as an array, with shape
            `(input_rows, input_cols, 3)`.
        theta (float): Angle to rotate our image by, in radians.

    Returns:
        (np.ndarray): Rotated image, with the same shape as the input.
    """
    input_rows, input_cols, channels = input_image.shape
    assert channels == 3

    center_x = input_rows // 2
    center_y = input_cols // 2

    # 1. Create an output image with the same shape as the input
    output_image = np.zeros_like(input_image)
    # 2. Loop over each pixel in the input image
    for i in range(input_rows):
        for j in range(input_cols):
            x_shifted = i - center_x
            y_shifted = j - center_y

            # 3. Get the coordinates of the current pixel in the input image
            input_i = int(round(center_x+(x_shifted * np.cos(theta) - y_shifted * np.sin(theta))))
            input_j = int(round(center_y+(x_shifted * np.sin(theta) + y_shifted * np.cos(theta))))
            if 0 <= input_i < input_rows and 0 <= input_j < input_cols:
                # 4. if the pixel is within bounds, copy it
                output_image[i, j , :] = input_image[input_i, input_j, :]
            else:
                # 5. if the pixel is out of bounds, set it to black
                output_image[i, j, :] = [0, 0, 0]

    # 3. Return the output image
    return output_image
