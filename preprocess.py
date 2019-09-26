
import os
import cv2
import pandas as pd
import multiprocessing as mp

root_dir = os.getcwd()
file_list = ['train.csv', 'val.csv']
image_source_dir = os.path.join(root_dir, 'data/images/')
data_root = os.path.join(root_dir, 'data')

def preprocess_image(image):
    # open image file
    img = cv2.imread(os.path.join(image_source_dir, image))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # perform transformations on image
    b = cv2.distanceTransform(img, distanceType=cv2.DIST_L2, maskSize=5)
    g = cv2.distanceTransform(img, distanceType=cv2.DIST_L1, maskSize=5)
    r = cv2.distanceTransform(img, distanceType=cv2.DIST_C, maskSize=5)
    
    # merge the transformed channels back to an image
    transformed_image = cv2.merge((b, g, r))
    target_file = os.path.join(image_target_dir, image)
    print("Writing target file {}".format(target_file))
    cv2.imwrite(target_file, transformed_image)

for file in file_list:
    
    image_target_dir = os.path.join(data_root, file.split(".")[0])
    
    if not os.path.exists(image_target_dir):
        os.mkdir(image_target_dir)
    
    # read list of image files to process from file
    print(os.path.join(data_root, file))
    image_list = pd.read_csv(os.path.join(data_root, file))['image_id']
    
    print("Start preprocessing images")
    pool = mp.Pool(processes=16)
    pool.map(preprocess_image, image_list)
print("Finished preprocessing images")
  
