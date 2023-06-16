import numpy as np
import cv2

def insert_bg(image: np.ndarray, bg: np.ndarray, polygon: list, h: int, w: int):
    """
    Inserts the foreground image onto the background image within the specified polygon region.

    Args:
        image (numpy.ndarray): Foreground image.
        bg (numpy.ndarray): Background image.
        polygon (list): List of polygon vertices [(x1, y1), (x2, y2), ...].
        h (int): Height of the output image.
        w (int): Width of the output image.

    Returns:
        numpy.ndarray: Output image with the foreground inserted in the specified region.
    """
    polygon=np.array([
            [int(x),int(y)]
            for (x, y) in polygon
        ])
    image=cv2.resize(image,(w,h))
    bg=cv2.resize(bg,(w,h))

    mask = np.zeros((h,w,3), dtype=np.uint8)
    cv2.fillPoly(mask,pts=[polygon],color=(1,1,1))

    only_object=image*mask
    bg=cv2.fillPoly(bg,pts=[polygon],color=(0,0,0))

    return only_object+bg

def translate_image(image_path: str, old_polygon: list, new_polygon: list, w: int, h: int):
    """
    Translates the input image and its polygon region to a new position and scale.

    Args:
        image_path (str): Path to the input image.
        old_polygon (list): List of polygon vertices in the input image [(x1, y1), (x2, y2), ...].
        new_polygon (list): List of polygon vertices in the output image [(x1, y1), (x2, y2), ...].
        w (int): Width of the output image.
        h (int): Height of the output image.

    Returns:
        tuple: A tuple containing the translated image and the updated polygon in the output image.
    """
    image1 = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
    h1, w1, _ = image1.shape
    old_polygon = np.array([[point[0]/w1*w, point[1]/h1*h] for point in old_polygon]).astype(np.int32)
    image1 = cv2.resize(image1, (w, h))
    
    edge_length1 = np.linalg.norm(old_polygon[0] - old_polygon[1])
    edge_length2 = np.linalg.norm(new_polygon[0] - new_polygon[1])
    scale = edge_length2/edge_length1
    move_x, move_y = new_polygon[0] - old_polygon[0]/edge_length1*edge_length2

    # Create the transformation matrix
    M = np.array([[scale, 0, move_x],
              [0, scale, move_y]], dtype=np.float32)
    # Apply the affine transformation
    translated_image = cv2.warpAffine(image1, M, (w, h))
    last_polygon = old_polygon.astype(np.int32)/edge_length1*edge_length2 + [move_x, move_y]
    return translated_image, last_polygon