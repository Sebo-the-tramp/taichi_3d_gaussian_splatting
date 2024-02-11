import os
import json
import random

import pandas as pd

# Function to read the entire JSON once and return needed parts as DataFrames
def extrapolate_json(file):
    """
    Load data from a JSON file and convert specific parts into pandas DataFrames.

    This function specifically targets a JSON structure that includes 'structure',
    'views', 'poses', and 'intrinsics' as keys. It assumes these keys contain lists
    that can be directly converted into pandas DataFrames.

    Parameters:
    - file (str): The path to the JSON file to be processed.

    Returns:
    - tuple: A tuple containing four pandas DataFrames:
        - structure: DataFrame with columns ['x', 'y', 'z'] representing points in space.
        - views: DataFrame indexed by 'poseId' with data from the 'views' part of the JSON.
        - poses: DataFrame with data from the 'poses' part of the JSON.
        - intrinsics: DataFrame with data from the 'intrinsics' part of the JSON.
    """

    with open(file) as f:
        data = json.load(f)
    
    # Check if the expected parts are lists and convert them directly to DataFrames    
    structure = pd.DataFrame([x['X'] for x in data['structure']], columns=["x", "y", "z"]).astype(float)        
    views = pd.DataFrame(data['views']).set_index('poseId')            
    poses = pd.DataFrame(data['poses'])            
    intrinsics = pd.DataFrame(data['intrinsics'])    
    return structure, views, poses, intrinsics

def prepare_intrinsic(intrinsic):        
    """
    Prepare the intrinsic camera matrix from the provided intrinsic parameters.

    Parameters:
    - intrinsic (dict): A dictionary containing the intrinsic parameters of the camera.
      Expected keys are 'principalPoint' and 'focalLength'.

    Returns:
    - list of lists: A 3x3 camera intrinsic matrix in the form:
      [[f, s, cx],
       [0, f, cy],
       [0, 0,  1]]
      where f is the focal length, (cx, cy) is the principal point, and s is the skew coefficient, assumed to be 0.
    """

    cx = float(intrinsic["principalPoint"][0][0])
    cy = float(intrinsic["principalPoint"][0][1])
    f = float(intrinsic["focalLength"][0])
    s = 0    

    return [[f,s,cx],
            [0,f,cy],
            [0,0,1]]

def prepare_rototranslation(original):    
    """
    Convert the rotation and translation data from a given format into a 4x4 rototranslation matrix.

    This function processes a dictionary that represents a camera pose, including its rotation
    and center (translation), and converts this information into a 4x4 matrix representing
    the rototranslation of the camera.

    Parameters:
    - original (dict): A dictionary containing the camera pose data. The key 'pose' should include
      another dictionary with 'transform' that has 'rotation' and 'center' as keys.

    Returns:
    - list of lists: A 4x4 rototranslation matrix combining both rotation and translation of the camera
      in the form of a homogeneous transformation matrix.
    """

    data = json.loads(str(original["pose"]).replace("'", "\""))

    rotation = data["transform"]['rotation']
    transform = data["transform"]['center']

    #convert to float
    rotation = [float(i) for i in rotation]
    transform = [float(i) for i in transform]    

    return [
        [rotation[0], rotation[1], rotation[2], transform[0]],
        [rotation[3], rotation[4], rotation[5], transform[1]],
        [rotation[6], rotation[7], rotation[8], transform[2]],        
        [0,0,0,1]
    ]

def convert_sfm_to_parquet():

    # Use the function to read data
    pd_structure, images, poses, intrinsic = extrapolate_json('sfm.json')

    # create the .parquet file
    point_cloud_df = pd.DataFrame(pd_structure["X"].to_list(), columns=["x", "y", "z"]).astype(float)
    point_cloud_df.to_parquet(os.path.join("./", "point_cloud.parquet"))

    # iterate for each image
    list_images = []

    for i in poses.iterrows():    
        new_pose = {}    
        id = i[1]["poseId"]   

        new_pose["image_path"] = images.loc[id]["path"]
        new_pose["T_pointcloud_camera"] = prepare_rototranslation(i[1])    
        new_pose["camera_intrinsics"] = prepare_intrinsic(intrinsic)
        new_pose["camera_height"] = float(images.loc[id]["height"])
        new_pose["camera_width"] = float(images.loc[id]["width"])
        new_pose["camera_id"] = 0    

        list_images.append(new_pose)            

    # create a sublist with 20 percent as validation

    random.shuffle(list_images)
    split = int(len(list_images) * 0.8)

    train = list_images[:split]
    val = list_images[split:]

    with open('train.json', 'w') as f:
        json.dump(train, f)

    with open('val.json', 'w') as f:
        json.dump(val, f)

if __name__ == "__main__":
    convert_sfm_to_parquet()