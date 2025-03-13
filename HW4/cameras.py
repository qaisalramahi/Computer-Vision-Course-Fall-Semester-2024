from typing import Tuple

import numpy as np


def camera_from_world_transform(d: float = 1.0) -> np.ndarray:
    """Define a transformation matrix in homogeneous coordinates that
    transforms coordinates from world space to camera space, according
    to the coordinate systems in Question 1.


    Args:
        d (float, optional): Total distance of displacement between world and camera
            origins. Will always be greater than or equal to zero. Defaults to 1.0.

    Returns:
        T (np.ndarray): Left-hand transformation matrix, such that c = Tw
            for world coordinate w and camera coordinate c as column vectors.
            Shape = (4,4) where 4 means 3D+1 for homogeneous.
    """
    theta = 3* (np.pi /4)

    # Rotation matrix around Z-axis
    R = np.array([
        [np.cos(theta), 0, np.sin(theta)],
        [0, 1, 0],
        [-np.sin(theta), 0, np.cos(theta)]
    ])

    # Translation vector from world to camera
    t = np.array([0, 0, d])

    # Homogeneous transformation matrix
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    
    assert T.shape == (4, 4)
    return T


def apply_transform(T: np.ndarray, points: np.ndarray) -> Tuple[np.ndarray]:
    """Apply a transformation matrix to a set of points.

    Hint: You'll want to first convert all of the points to homogeneous coordinates.
    Each point in the (3,N) shape edges is a length 3 vector for x, y, and z, so
    appending a 1 after z to each point will make this homogeneous coordinates.

    You shouldn't need any loops for this function.

    Args:
        T (np.ndarray):
            Left-hand transformation matrix, such that c = Tw
                for world coordinate w and camera coordinate c as column vectors.
            Shape = (4,4) where 4 means 3D+1 for homogeneous.
        points (np.ndarray):
            Shape = (3,N) where 3 means 3D and N is the number of points to transform.

    Returns:
        points_transformed (np.ndarray):
            Transformed points.
            Shape = (3,N) where 3 means 3D and N is the number of points.
    """
    N = points.shape[1]
    assert points.shape == (3, N)

    points_homogenous = np.vstack([points,np.ones((1,N))])

    points_transformed_homogenous = T @ points_homogenous

    points_transformed = points_transformed_homogenous[:3,:]

    assert points_transformed.shape == (3, N)
    return points_transformed


def intersection_from_lines(
    a_0: np.ndarray, a_1: np.ndarray, b_0: np.ndarray, b_1: np.ndarray
) -> np.ndarray:
    """Find the intersection of two lines (infinite length), each defined by a
    pair of points.

    Args:
        a_0 (np.ndarray): First point of first line; shape `(2,)`.
        a_1 (np.ndarray): Second point of first line; shape `(2,)`.
        b_0 (np.ndarray): First point of second line; shape `(2,)`.
        b_1 (np.ndarray): Second point of second line; shape `(2,)`.

    Returns:
        np.ndarray: the intersection of the two lines definied by (a0, a1)
                    and (b0, b1).
    """
    # Validate inputs
    assert a_0.shape == a_1.shape == b_0.shape == b_1.shape == (2,)
    assert a_0.dtype == a_1.dtype == b_0.dtype == b_1.dtype == float

    # Intersection point between lines
    out = np.zeros(2)
    
    # Slope Calculation
    m1 = (a_1[1] - a_0[1]) / (a_1[0] - a_0[0])
    m2 = (b_1[1] - b_0[1]) / (b_1[0] - b_0[0])

    # Constant calculation 
    c1 = a_0[1] - (m1*a_0[0])
    c2 = b_0[1] - (m2*b_0[0])

    # Final substitution 
    out[0] = (c2-c1) / (m1-m2)
    out[1] = m1 * out[0] + c1

    assert out.shape == (2,)
    assert out.dtype == float

    return out


def optical_center_from_vanishing_points(
    v0: np.ndarray, v1: np.ndarray, v2: np.ndarray
) -> np.ndarray:
    """Compute the optical center of our camera intrinsics from three vanishing
    points corresponding to mutually orthogonal directions.

    Hints:
    - Your `intersection_from_lines()` implementation might be helpful here.
    - It might be worth reviewing vector projection with dot products.

    Args:
        v0 (np.ndarray): Vanishing point in image space; shape `(2,)`.
        v1 (np.ndarray): Vanishing point in image space; shape `(2,)`.
        v2 (np.ndarray): Vanishing point in image space; shape `(2,)`.

    Returns:
        np.ndarray: Optical center; shape `(2,)`.
    """
    assert v0.shape == v1.shape == v2.shape == (2,), "Wrong shape!"

    optical_center = np.zeros(2)

    # slope of side of triangle 
    slope_side1 = (v2[1] - v1[1]) / (v2[0] - v1[0])
    # slope of altitude line which is perpendicular to 2 vps and passing through the 3rd one+
    slope_altitude1 = - (1 / slope_side1)
    # intercept of altitude
    intercept1 = v0[1] - slope_altitude1 * v0[0]

     # Compute the slope and intercept of the altitude through v1
    slope_side2 = (v2[1] - v0[1]) / (v2[0] - v0[0])  # Slope of side v0-v2
    slope_altitude2 = -1 / slope_side2  
    intercept2 = v1[1] - slope_altitude2 * v1[0]  

    # Solve for the intersection of the two altitudes
    # wasnt sure how to reuse intersection_of_lines function since it took in 4 parameters
    x_optical = (intercept2 - intercept1) / (slope_altitude1 - slope_altitude2)
    y_optical = slope_altitude1 * x_optical + intercept1

    optical_center = np.array([x_optical, y_optical])
    assert optical_center.shape == (2,)
    return optical_center


def focal_length_from_two_vanishing_points(
    v0: np.ndarray, v1: np.ndarray, optical_center: np.ndarray
) -> np.ndarray:
    """Compute focal length of camera, from two vanishing points and the
    calibrated optical center.

    Args:
        v0 (np.ndarray): Vanishing point in image space; shape `(2,)`.
        v1 (np.ndarray): Vanishing point in image space; shape `(2,)`.
        optical_center (np.ndarray): Calibrated optical center; shape `(2,)`.

    Returns:
        float: Calibrated focal length.
    """
    assert v0.shape == v1.shape == optical_center.shape == (2,), "Wrong shape!"

    x0, y0 = v0
    x1,y1 = v1
    cx,cy = optical_center
    f_squared = -((x0-cx)*(x1-cx) + (y0-cy)*(y1-cy))

    assert f_squared > 0 

    f = np.sqrt(f_squared)

    return float(f)


def physical_focal_length_from_calibration(
    f: float, sensor_diagonal_mm: float, image_diagonal_pixels: float
) -> float:
    """Compute the physical focal length of our camera, in millimeters.

    Args:
        f (float): Calibrated focal length, using pixel units.
        sensor_diagonal_mm (float): Length across the diagonal of our camera
            sensor, in millimeters.
        image_diagonal_pixels (float): Length across the diagonal of the
            calibration image, in pixels.

    Returns:
        float: Calibrated focal length, in millimeters.
    """
    pixel_to_mm = sensor_diagonal_mm / image_diagonal_pixels

    f_mm = f * pixel_to_mm

    return f_mm
