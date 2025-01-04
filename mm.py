import os
import cv2
import numpy as np




def extract(image):
    img = image
    image_copy = image
    factor=0.5
    factor = max(0, min(1, factor))
    b, g, r = cv2.split(image)                    
    b = np.clip(b * factor, 0, 255).astype(np.uint8)
    g = np.clip(g * factor, 0, 255).astype(np.uint8)
    r = np.clip(r * factor, 0, 255).astype(np.uint8)
    image = cv2.merge([b, g, r])                      
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_blue_hue = 100
    upper_blue_hue = 140
    blue_mask = cv2.inRange(hsv_image, (lower_blue_hue, 50, 50), (upper_blue_hue, 255, 255))
    blue_mask = cv2.merge([blue_mask, blue_mask, blue_mask])
    white_image = np.ones_like(image) * 255
    result_image = np.where(blue_mask > 0, image_copy, white_image)
    #combined_image = np.concatenate((img, result_image), axis=1)
    
    
    return result_image
    
    
input_directory = "document_data"

# Path to the directory where processed images will be saved
output_directory = "new_doc"

# Ensure the output directory exists
os.makedirs(output_directory, exist_ok=True)

# Iterate over each file in the input directory
for filename in os.listdir(input_directory):
    image_path = os.path.join(input_directory, filename)
    image = cv2.imread(image_path)
    

    processed_image = extract(image)
   
    # Concatenate original and processed images horizontally
    combined_image = np.concatenate((image, processed_image), axis=1)
    
    # Save the processed image to the output directory
    output_path = os.path.join(output_directory, filename)
    cv2.imwrite(output_path, combined_image)  
