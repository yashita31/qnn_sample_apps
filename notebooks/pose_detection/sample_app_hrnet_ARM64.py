#!/usr/bin/env python
# coding: utf-8

# ![image.png](attachment:b9700005-fdce-4534-9d0e-b69af6531556.png)

# ## Python Windows ARM64

# In[1]:


# !powershell pip list


# In[2]:


import platform

arch = platform.machine()
sys = platform.system()
processor = platform.processor()
print(f"{arch}\n{sys}\n{processor}")


# In[3]:


import cv2 as cv
import numpy as np
import onnxruntime as ort

from PIL import Image
from pathlib import Path
from typing import List, Tuple


# In[4]:


root_dir = Path.cwd().parent
onnxruntime_dir = Path(ort.__file__).parent


# In[5]:


model_subdirectory = "hrnet_pose"
model_name = "hrnet_quantized.onnx"
model_path = Path.joinpath(root_dir,"models",model_subdirectory,model_name) 

hexagon_driver = Path.joinpath(onnxruntime_dir,"capi","QnnHtp.dll")

qnn_provider_options = {
    "backend_path": hexagon_driver,
}

session = ort.InferenceSession(model_path, 
                               providers= [("QNNExecutionProvider",qnn_provider_options),"CPUExecutionProvider"],
                              )
## Retrieve expected input from model
inputs = session.get_inputs()
outputs = session.get_outputs()
input_0 = inputs[0] 
output_0 = outputs[0]
session.get_providers()


# In[6]:


print(f"Expected Input Shape: {input_0.shape}")
print(f"Expected Input Type: {input_0.type}")
print(f"Expected Input Name: {input_0.name}")


# In[7]:


print(f"Expected Output Shape: {output_0.shape}")
print(f"Expected Output Type: {output_0.type}")
print(f"Expected Output Name: {output_0.name}")


# In[8]:


meta_data = session.get_modelmeta()
meta_data.custom_metadata_map


# <h4>Input Frame needs to be transformed to the shape and datatype below:</h4>
# 
# 1. **Shape: (1,3,256,192) => (B,C,H,W)**
# 2. **Datatype: Float 32**
# 3. **Name: Image**

# In[9]:


expected_shape = input_0.shape

def transform_numpy_opencv(image: np.ndarray, 
                           expected_shape
                          ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Resize and normalize an image using OpenCV, and return both HWC and CHW formats.

    Parameters:
    -----------
    image : np.ndarray
        Input image in HWC (Height, Width, Channels) format with dtype uint8.
    
    expected_shape : tuple or list
        Expected shape of the model input, typically in the format (N, C, H, W).
        Only the height and width (H, W) are used for resizing.

    Returns:
    --------
    tuple of np.ndarray
        - float_image: The resized and normalized image in HWC format (float32, range [0, 1]).
        - chw_image: The same image converted to CHW format, suitable for deep learning models.
    """
    
    height, width = expected_shape[2], expected_shape[3]
    resized_image = cv.resize(image, (width, height), interpolation=cv.INTER_CUBIC)
    float_image = resized_image.astype(np.float32) / 255.0
    chw_image = np.transpose(float_image, (2,0,1)) # HWC -> CHW

    return (float_image,chw_image)


# In[10]:


def keypoint_processor_numpy(post_inference_array: np.ndarray, 
                             scaler_height: int, 
                             scaler_width: int
                            ) -> List[Tuple[int, int]]:
    """
    Extracts keypoint coordinates from heatmaps and scales them to match the original image dimensions.

    Parameters:
    -----------
    post_inference_array : np.ndarray
        A 3D array of shape (num_keypoints, heatmap_height, heatmap_width),
        containing the model's predicted heatmaps for each keypoint.
    
    scaler_height : int
        Scaling factor for the height dimension to map from heatmap space to original image space.
    
    scaler_width : int
        Scaling factor for the width dimension to map from heatmap space to original image space.

    Returns:
    --------
    list of tuple
        A list of (y, x) coordinates (as integers) representing the scaled keypoint positions
        in the original image space.
    """
    keypoint_coordinates = []

    for keypoint in range(post_inference_array.shape[0]):
        heatmap = post_inference_array[keypoint]
        max_val_index = np.argmax(heatmap)
        img_height, img_width = np.unravel_index(max_val_index, heatmap.shape)
        coords = (int(img_height * scaler_height), int(img_width * scaler_width))
        keypoint_coordinates.append(coords)

    return keypoint_coordinates
        


# In[11]:


# 0: System Camera
cap = cv.VideoCapture(0)

if not cap.isOpened():
    print("Invalid Camera Selected")
    exit()

###########################################################################
## This is for scaling purposes ###########################################
###########################################################################
input_image_height, input_image_width = expected_shape[2], expected_shape[3]

heatmap_height, heatmap_width = 64, 48
scaler_height = input_image_height/heatmap_height
scaler_width = input_image_width/heatmap_width

while True:

    ret, hwc_frame = cap.read()
 
    if not ret:
        print("Can't receive frame (stream end?). Exiting...")
        break

    hwc_frame_processed, chw_frame = transform_numpy_opencv(hwc_frame, expected_shape)

    ########################################################################
    ## INFERENCE ###########################################################
    ########################################################################
    inference_frame = np.expand_dims(chw_frame, axis=0)
    outputs = session.run(None, {input_0.name:inference_frame})

    output_tensor = np.array(outputs).squeeze(0).squeeze(0)
    
    keypoint_coordinate_list = keypoint_processor_numpy(output_tensor, scaler_height, scaler_width)

    ########################################################################
    # SCALE AND MAP KEYPOINTS BACK TO ORIGINAL FRAME THEN DISPLAY THAT FRAME
    ########################################################################
    frame = (hwc_frame_processed*255).astype(np.uint8)
    frame = frame.copy()

    for (y,x) in keypoint_coordinate_list:
        cv.circle(frame, (x,y), radius=3, color=(0,0,255), thickness=-1)
        
    frame = cv.resize(frame, (640,480), interpolation=cv.INTER_CUBIC)    
    cv.imshow('frame',frame)
    if cv.waitKey(1) == ord('q'):
        break

cap.release()
cv.destroyAllWindows()


# In[ ]:




