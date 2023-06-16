import json
import os
import math
import numpy as np
import random
from shapely.geometry import Polygon

def polygon_size(polygon: np.ndarray):
    """
    Calculate the width and height of a polygon.

    Args:
        polygon (numpy.ndarray): Array representing the polygon vertices.

    Returns:
        tuple: Width and height of the polygon.
    """
    x_coordinates = polygon[:, 0]  # Extract the x-coordinates from the polygon vertices
    y_coordinates = polygon[:, 1]  # Extract the y-coordinates from the polygon vertices

    # Calculate the height and width
    height = np.max(y_coordinates) - np.min(y_coordinates)
    width = np.max(x_coordinates) - np.min(x_coordinates)

    return width, height
def poly_intersect(poly1, poly2):
    """
    Check if two polygons intersect.

    Args:
        poly1 (list): List of vertices of the first polygon.
        poly2 (list): List of vertices of the second polygon.

    Returns:
        bool: True if the polygons intersect, False otherwise.
    """
    shapely_polygon1 = Polygon(poly1)
    shapely_polygon2 = Polygon(poly2)

    # Check if the polygons intersect
    if shapely_polygon1.intersects(shapely_polygon2):
        return True
    return False

def polygon_move(polygon: np.ndarray, move_x: float, move_y: float):
    """
    Move a polygon by a given displacement.
    Args:
        polygon (numpy.ndarray): Array representing the polygon vertices.
        move_x (float): Amount to move in the x-axis.
        move_y (float): Amount to move in the y-axis.

    Returns:
        numpy.ndarray: Array representing the moved polygon.
    """
    polygon[:, 0] += move_x
    polygon[:, 1] += move_y
    return polygon

def gen_polygon(polygon: list, new_polygon_list: list[np.ndarray], w: int, h: int) -> np.ndarray:
    """
    Generate a new polygon by randomly positioning the input polygon within the specified dimensions.

    Args:
        polygon (list): List of vertices of the input polygon.
        new_polygon_list (list[numpy.ndarray]): List of existing polygons in the new image.
        w (int): Width of the new image.
        h (int): Height of the new image.

    Returns:
        numpy.ndarray or False: New polygon if generated successfully, False otherwise.
    """
    polygon = np.array(polygon, dtype= "float32")
    polygon = polygon_move(polygon, -np.min(polygon[:, 0]), -np.min(polygon[:, 1]))

    poly_w, poly_h = polygon_size(polygon)
    if w-poly_w <=0 or h-poly_h <=0:
        return False
    poly_pos = [
        np.random.randint(0, w-poly_w),
        np.random.randint(0, h-poly_h)
    ]
    copy_polygon = polygon.copy()
    new_polygon = polygon_move(copy_polygon, poly_pos[0], poly_pos[1])
    intersect = True
    if new_polygon_list == []:
        return new_polygon
    count_loop = 0
    while intersect:
        intersect = False
        for old_polygon in new_polygon_list:
            # if cv2.intersectConvexConvex(new_polygon, old_polygon)[0] > 0:
            if poly_intersect(new_polygon, old_polygon):
                intersect = True
                count_loop+=1
                break
        if intersect:
            poly_pos = [
                np.random.randint(0, w-poly_w),
                np.random.randint(0, h-poly_h)
            ]
            copy_polygon = polygon.copy()
            new_polygon = polygon_move(copy_polygon, poly_pos[0], poly_pos[1])
        if count_loop == 50:
            return False
    return new_polygon

def multi_image_augment(labelme_folder: str, json_list: list, num_img = 2):
    """
    Perform image augmentation by randomly selecting and transforming images.

    Args:
        labelme_folder (str): Path to the folder containing the LabelMe JSON files.
        json_list (list): List of JSON filenames.
        num_img (int): Number of images to augment.

    Returns:
        tuple: A tuple containing the new polygon list, width, height, and paths to the selected JSON files.
    """
    
    filenames = random.choices(json_list, k=num_img)
    json_paths = [os.path.join(labelme_folder, filename) for filename in filenames]

    w_list = []
    h_list = []
    polygon_list = []

    for json_path in json_paths:
        with open(json_path, "r") as f:
            json_data = json.load(f)
            w = json_data['imageWidth']
            h = json_data['imageHeight']
            w_list.append(w)
            h_list.append(h)
            polygon = json_data['shapes'][0]['points']
            new_polygon = [[point[0]/w, point[1]/h] for point in polygon]
            polygon_list.append(new_polygon)

    max_w = max(w_list)
    max_h = max(h_list)
    for i, polygon in enumerate(polygon_list):
        new_polygon = [[point[0]*max_w, point[1]*max_h] for point in polygon]
        polygon_list[i] = new_polygon
        
    new_area = max_w*max_h*num_img
    new_w = int(math.sqrt(new_area))
    new_h = int(max_h/max_w*new_w)

    new_polygon_list=[]
    for polygon in polygon_list:
        new_polygon = gen_polygon(polygon, new_polygon_list, new_w, new_h)
        while type(new_polygon) == bool:
            new_w = int(new_w*1.2)
            new_h = int(new_h*1.2)
            new_polygon = gen_polygon(polygon, new_polygon_list, new_w, new_h)
        new_polygon_list.append(new_polygon)
        
    return new_polygon_list, new_w, new_h, json_paths