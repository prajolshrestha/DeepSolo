import os
import  cv2
import json
import numpy as np
import random 
import math
import logging
import time

from .curve_utils import BezierCurve

from detectron2.structures import BoxMode
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.utils.visualizer import Visualizer

logger = logging.getLogger(__name__)

__all__ = ["load_map_text_json", "register_map_text_instances"]


def register_map_text_instances(name, metadata, json_file, image_root, voc_size_cfg, num_pts_cfg):
    """
    Register a dataset in json annotation format for text detection and recognition.

    Args:
        name (str): a name that identifies the dataset, e.g. "lvis_v0.5_train".
        metadata (dict): extra metadata associated with this dataset. It can be an empty dict.
        json_file (str): path to the json instance annotation file.
        image_root (str or path-like): directory which contains all the images.
    """
    DatasetCatalog.register(name, lambda: load_map_text_json(json_file, image_root, name, voc_size_cfg=voc_size_cfg, num_pts_cfg=num_pts_cfg))
    MetadataCatalog.get(name).set(json_file=json_file, image_root=image_root, evaluator_type="text", **metadata)

   

def load_map_text_json(json_file, image_root, dataset_name=None, extra_annotation_keys=None, voc_size_cfg=96, num_pts_cfg=25):
    """
        Args:
            json_file (str): full path to the json file in totaltext annotation format.
            image_root (str or path-like): the directory where the images in this json file exists.
        
        Returns:
            list[dict]: a list of dicts in Detectron2 standard dataset dicts format.
    """
    dataset_dicts = [] # to store info about each image in dataset

    # Customized for winnman dataset. Any other way??
    number_of_img = 25
    if image_root == "datasets/maps/val":
        number_of_img = 6

    logger.info(f"Map data registration from {image_root} started! Usually takes some moment...")
    

    for idx in range(number_of_img):
        # Load single json file corresponding to the image file
        json_file_path = os.path.join(json_file,f"image{idx}.json")
        with open(json_file_path) as f:
            imgs_anns = json.load(f) 
    
        ## Images
        record = {} 
        
        filename= os.path.join(image_root,f"image{idx}.tiff")
        height, width = cv2.imread(filename).shape[:2]
        
        record["file_name"] = filename
        record["height"] = height
        record["width"] = width
        record["image_id"] = idx 
        
        print(f'Image {idx} registration started!')
        start_time = time.time()

        ## Annotations
        objs = [] 
        for _, anno in enumerate(imgs_anns):
            ## TEXT
            text = anno["text"]
            # out of bound text and char that leads to cost matrix error!
            # if voc_size_cfg == 96 and text in (None, "\u00a9", "ꟼ", "°", "é"):
            #     continue
            # if voc_size_cfg == 37 and text in (None, "\u00a9", "ꟼ", "°", "&", "é"):
            #     continue

            if text is None:
                continue
            # text to integer
            unicode_integer = text_to_int(text, voc_size_cfg)
            # Filter out of bound text instances
            if voc_size_cfg == 37 and unicode_integer == [37] * num_pts_cfg:
                continue
            if voc_size_cfg == 96 and unicode_integer == [96] * num_pts_cfg:
                continue

            ##bbox
            items = anno["items"]
            text_coordinates = [segment["points"] for segment in anno["items"]] # collect all the points of a word from different segments
            px = [point[0] for segment in text_coordinates for point in segment] # all x coordinates of a word
            py = [point[1] for segment in text_coordinates for point in segment] # all y coordinates of a word

            ## center beziers, boundary, polyline
            lower_pts = []
            upper_pts = []
            # case 1: Characters are coherent
            if (len(items)  == 1): 
                pts = items[0]["points"] 
                lower_pt, upper_pt = compute_lower_and_upper_pts(pts)
                lower_pts = np.array(lower_pt)
                upper_pts =np.array(upper_pt[::-1])

            # case 2: characters are split up
            else: 
                for segment in items:
                    pts = segment["points"]
                    lower_pt, upper_pt = compute_lower_and_upper_pts(pts)
                    lower_pts.extend(lower_pt)
                    upper_pts.extend(upper_pt[::-1])
                lower_pts = np.array(lower_pts)
                upper_pts = np.array(upper_pts) # this is different

            # Boundary and polyline
            lower_bezierpts, sample_points_lower = compute_bezier(lower_pts,n=25)
            upper_bezierpts, sample_points_upper = compute_bezier(upper_pts,n=25)
            boundary = np.concatenate((sample_points_lower, sample_points_upper[::-1]), axis = 0)  
            polyline = (sample_points_lower + sample_points_upper) / 2

            # center_bezierpoints from polyline
            centerpoints = polyline 
            center_bezierpts, sample_points_center = compute_bezier(centerpoints)

   
            obj = {
                "iscrowd": 0, 
                "category_id": 0,  
                "bbox": [np.min(px), np.min(py), np.max(px), np.max(py)],
                "bbox_mode": BoxMode.XYXY_ABS,
                "beziers": center_bezierpts,
                "boundary": boundary,
                "polyline": polyline,
                "text": unicode_integer,
                "word": text # only for testing purpose  
            }
            objs.append(obj)
            #break
                      
        record["annotations"] = objs
        dataset_dicts.append(record)   

        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f'Image {idx} registration completed in {elapsed_time} seconds!')

    
    return dataset_dicts

def compute_lower_and_upper_pts(pts):
    if len(pts) == 4: # Why not measure angle for all case? => Most of the cases has 4 points! This way is less error prone!
        lower_pts = pts[:2]
        upper_pts = pts[2:] 

    else: # How to deal with more than 4 points?? ==> Find right bottom corner. => find angle!
        points = [ coord for index, coord in enumerate(pts) if coord not in pts[:index]] # Removes duplicate coords
        for i in range(len(points) - 2):
            v1 = np.array(points[i + 1]) - np.array(points[i])
            v2 = np.array(points[i + 2]) - np.array(points[i + 1])
            dot_product = np.dot(v1, v2)
            v1_magnitude = np.linalg.norm(v1)
            v2_magnitude = np.linalg.norm(v2)
            
            similarity = dot_product / (v1_magnitude * v2_magnitude)
            clamped_similarity = np.clip(similarity, -1, 1) # cos range [-1, 1]    
            angle = math.degrees(math.acos(clamped_similarity))
            if angle > 45: # What is the correct angle that differentiates corner and st. line ?
                last_lower_point_idx = i+1
                break
    
        lower_pts = points[:last_lower_point_idx+1]
        upper_pts = points[last_lower_point_idx+1:]

    return lower_pts, upper_pts

def compute_bezier(points,n=50):
    points = np.array(points)
    bc = BezierCurve(order=3)
    bc.get_control_points(points[:, 0], points[:, 1], interpolate=True)
    control_pts = bc.control_points 
    control_pts = np.array(control_pts)
    sample_pts = bc.get_sample_point(n)

    return control_pts, sample_pts

def text_to_int(text, voc_size):
    if voc_size == 96:
        vocabulary = [' ','!','"','#','$','%','&','\'','(',')','*','+',',','-','.','/','0','1','2','3','4','5','6','7','8','9',':',';','<','=','>','?','@','A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z','[','\\',']','^','_','`','a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z','{','|','}','~']
        char_to_int = lambda char: vocabulary.index(char) if char in vocabulary else voc_size
        integer_list = [char_to_int(char) for char in text]
        
    elif voc_size == 37: # a-z + 0-9 + unknown
        vocabulary = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z','0','1','2','3','4','5','6','7','8','9']
        char_to_int = lambda char: vocabulary.index(char)  if char in vocabulary else voc_size
        integer_list = [char_to_int(char) for char in text.lower()]
    
    if len(integer_list) < num_pts_cfg:
        integer_list.extend([voc_size] * (num_pts_cfg - len(integer_list)))

    return integer_list[:num_pts_cfg]


###################################### Debug & Visualize ###################################################

def check_ground_truth(dataset_dict):

    out_of_bounds_instances = []
    for idx, image_data in enumerate(dataset_dict):
        for anno in image_data["annotations"]:
            text = anno["word"]
            #if len(text) == 1:# text with single instance
            for char in text:
                if voc_size_cfg == 37 and char.lower() not in ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z','0','1','2','3','4','5','6','7','8','9']:
                    out_of_bounds_instances.append((idx, char))
                elif voc_size_cfg == 96 and char not in [' ','!','"','#','$','%','&','\'','(',')','*','+',',','-','.','/','0','1','2','3','4','5','6','7','8','9',':',';','<','=','>','?','@','A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z','[','\\',']','^','_','`','a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z','{','|','}','~']:
                    out_of_bounds_instances.append((idx, char))
    

    if out_of_bounds_instances:
        print(f"Voc_size: {voc_size_cfg}")
        print("Out of bounds instances found: ")
        for idx, char in out_of_bounds_instances:
            print(f"Image {idx}, Char: {char}")
    else: 
        print("No out of bound instances found!")
        
def visualize_boundary_and_curve(dataset_dict):
    
    #Visualize boundary and polyline and bezier curve
    for d in random.sample(dataset_dict, 1):
        img = cv2.imread(d["file_name"])
        
        # Draw boundaries, Bezier curves, and polyline
        for ann in d["annotations"]:
            boundary = ann["boundary"].astype(np.int32)
            bezier_pts = ann["beziers"].astype(np.int32)
            polyline = ann["polyline"].astype(np.int32)

            # Draw boundary
            cv2.polylines(img, [boundary], isClosed=True, color=(255, 0, 0), thickness=2)

            # Draw Bezier curve
            for i in range(0, len(bezier_pts), 4):
                cv2.polylines(img, [bezier_pts[i:i+4]], isClosed=False, color=(0, 255, 0), thickness=2)

            # Draw polyline
            cv2.polylines(img, [polyline], isClosed=False, color=(0, 0, 255), thickness=2)

        print(f'Plotting {d["file_name"]}')
        cv2.imshow("image", img[:,:,::-1])
        cv2.waitKey(0)
        cv2.destroyAllWindows()

def visualize_bbox(dataset_dict):
    
    # Visualize annotation (BBox)
    map_metadata = MetadataCatalog.get("maps_val")
    for d in random.sample(dataset_dict, 1):
        img = cv2.imread(d["file_name"])
        visualizer = Visualizer(img[:,:,::-1], metadata=map_metadata, scale=0.5)
        out = visualizer.draw_dataset_dict(d) # random dict passed for output
        cv2.imshow("image",out.get_image()[:,:,::-1])
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    
    voc_size_cfg = 96 #37
    num_pts_cfg = 25

    #root = "datasets"
    json_file = 'maps/map-anno/train'
    image_root = 'maps/train'
    
    val = True #False
    if val:
        json_file = 'maps/map-anno/val'
        image_root = 'maps/val'
    
    # Register dataset
    #dataset_dict = load_map_text_json(os.path.join(root,json_file), os.path.join(root,image_root) , voc_size, num_pts_cfg)
    dataset_dict = load_map_text_json(json_file, image_root , voc_size_cfg, num_pts_cfg)

    check_ground_truth(dataset_dict)
    visualize_boundary_and_curve(dataset_dict)
    #visualize_bbox(dataset_dict)