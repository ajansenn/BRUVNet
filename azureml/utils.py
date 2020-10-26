from __future__ import print_function
from PIL import Image
import json
import random
import cv2
import os
import numpy as np
import json
from detectron2.structures import BoxMode
import itertools
from tqdm import tqdm

def image_dimensions(image_file_path):
    image = Image.open(image_file_path)
    img_width, img_height = image.size
    if image_file_path.lower().endswith("png"):
        img_depth = 1
    else:
        img_depth = image.layers
    return img_width, img_height, img_depth

# write a function that extracts labels from via
def get_labels_from_via(img_dir, json_file):
    species_list = {}
    json_file = os.path.join(img_dir, json_file)    
    imgs_anns = json.loads(open(json_file).read())
    imgs_anns = imgs_anns['_via_img_metadata']
    for _, v in imgs_anns.items():
        try:
            annos = v["regions"]
            objs = []
            for anno in annos:         
                # K.B. check species and add to dictionary for lookup
                if anno['region_attributes']['Fish Species'] not in species_list.keys():
                    species_id = len(species_list)
                    species_list.update({anno['region_attributes']['Fish Species'] : species_id })                              
        except AttributeError:
            #print("shape not found")
            pass
        #code to move to next frame
    return species_list

# write a function that loads the dataset into detectron2's standard format
#def get_fish_dicts(img_dir, json_file_path):
def get_fish_dicts(images_dir, json_file, class_labels):    
    #json_file = os.path.join(images_dir, json_file)
    with open(json_file) as f:
        imgs_anns = json.load(f)
        imgs_anns = imgs_anns['_via_img_metadata']
    dataset_dicts = []
    image_id = 0
    for _, v in imgs_anns.items():
        record = {}
        filename = os.path.join(images_dir, v["filename"])
        file_id = os.path.splitext(v["filename"])[0]
        try:

            height, width = cv2.imread(filename).shape[:2]
            record["file_name"] = filename
            record["height"] = height
            record["width"] = width
            # uncomment for semantic segmentation tasks
            #record["sem_seg_file_name"] = semseg_filename
            annos = v["regions"]
            objs = []

            #for _, anno in annos.items():
            for anno in annos:         
                anno_att = anno["shape_attributes"]
                px = anno_att["all_points_x"]
                py = anno_att["all_points_y"]
                poly = [(x + 0.5, y + 0.5) for x, y in zip(px, py)]
                poly = list(itertools.chain.from_iterable(poly))
                
                obj = {
                    "bbox": [np.min(px), np.min(py), np.max(px), np.max(py)],
                    "bbox_mode": BoxMode.XYXY_ABS,
                    "segmentation": [poly],
                    "category_id": 0,
                    "category_id": class_labels[anno['region_attributes']['Fish Species']],
                    #"category_id": species_list[anno['region_attributes']['Fish Species']],
                    "iscrowd": 0                    
                }
                objs.append(obj)
            record["annotations"] = objs
            record["image_id"] = image_id
            image_id += 1
            dataset_dicts.append(record)
        except AttributeError:
            #print("shape not found")
            pass
        #code to move to next frame
    return dataset_dicts

## Create masks for semantic segmentation
def convert_vgg_to_masks(images_dir, json_file, masks_dir, class_labels, combine_into_one):
    # check folder/files
    if not os.path.exists(images_dir):
        raise ValueError("Images directory does not exist")
    elif not os.path.exists(json_file):
        raise ValueError("VGG JSON not found")

    os.makedirs(masks_dir, exist_ok=True)

    # load the VGG JSON file 
    annotations = json.loads(open(json_file).read())
    annotations = annotations['_via_img_metadata']
    image_annotations = {}

    # loop over the file ID and annotations themselves (values)
    for data in annotations.values():
        # store the data in the dictionary using the filename as the key
        image_annotations[data["filename"]] = data

    # get a dictionary of class labels to class IDs
    #class_labels = _class_labels_to_ids(class_labels_file)
    
    print("Generating mask files...")
    for image_file_name in tqdm(os.listdir(images_dir)):

        # skip any files without an annotation
        #if not image_file_name.endswith(".jpg"):
        if not image_file_name in image_annotations.keys():
            continue

        file_id = os.path.splitext(image_file_name)[0]

        # grab the image info and then grab the annotation data for
        # the current image based on the unique image ID
        
        annotation = image_annotations[image_file_name]
        
#        print(f"working on file: {image_file_name}")
        # get the image's dimensions
        width, height, _ = image_dimensions(os.path.join(images_dir, image_file_name))

        # if combining all regions into a single mask file
        # then we'll only need to allocate the mask array once
        if combine_into_one:
            # allocate memory for the region mask
            region_mask = np.zeros((height, width, 3), dtype="uint8")

        # loop over each of the annotated regions
        for (i, region) in enumerate(annotation["regions"]):

            # if not combining all regions into a single mask file then
            # we'll need to reallocate the mask array for each mask region
            if not combine_into_one:
                # allocate memory for the region mask
                region_mask = np.zeros((height, width, 3), dtype="uint8")

            # grab the shape and region attributes
            shape_attributes = region["shape_attributes"]
            region_attributes = region["region_attributes"]

            # find the class ID corresponding to the region's class attribute
            #class_label = region_attributes["class"]
            class_label = region_attributes["Fish Species"]
            if class_label not in class_labels:
                raise ValueError(
                    "No corresponding class ID found for the class label "
                    f"found in the region attributes -- label: {class_label}",
                )
            else:
                class_id = class_labels[class_label]

            # get the array of (x, y)-coordinates for the region's mask polygon
            x_coords = shape_attributes["all_points_x"]
            y_coords = shape_attributes["all_points_y"]
            coords = zip(x_coords, y_coords)
            poly_coords = [[x, y] for x, y in coords]
            pts = np.array(poly_coords, np.int32)

            # reshape the points to (<# of coordinates>, 1, 2)
            pts = pts.reshape((-1, 1, 2))

            # draw the polygon mask, using the class ID as the mask value
            cv2.fillPoly(region_mask, [pts], color=[class_id]*3)

            # if not combining all masks into a single file
            # then write this mask into its own file
            if not combine_into_one:
                # write the mask file
                mask_file_name = f"{file_id}_segmentation_{i}.png"
                cv2.imwrite(os.path.join(masks_dir, mask_file_name), region_mask)

        # write a combined mask file, if requested
        if combine_into_one:
            # write the mask file
            mask_file_name = f"{file_id}_segmentation.png"
            cv2.imwrite(os.path.join(masks_dir, mask_file_name), region_mask)

def split_json_train_test(input_dir, source_json, output_dir, split_pct = 0.7): 
    annotations_file = os.path.join(input_dir, source_json)
    # arguments validation
    if not os.path.exists(annotations_file):
        raise ValueError(f"Invalid json file path: {annotations_file}")
    
    raw_json = json.loads(open(annotations_file).read())
    via_settings = raw_json['_via_settings']
    via_metadata = raw_json['_via_img_metadata']
    
    # work out random split
    key_len = len(via_metadata)
    rnd_keys = random.sample(list(via_metadata.keys()), key_len)
    split_idx = int(key_len * split_pct)
    train_key = rnd_keys[:split_idx]
    val_key = rnd_keys[split_idx:]

    train_dict = {key:via_metadata[key] for key in train_key}
    val_dict = {key:via_metadata[key] for key in val_key}
    
    updated_train_json = {}
    updated_train_json.update(via_settings)
    updated_train_json.update({"_via_img_metadata" :train_dict})
    
    with open(os.path.join(output_dir, "via_region_data_train.json"), 'w') as json_file:
        json.dump(updated_train_json, json_file)
    
    updated_val_json = {}
    updated_val_json.update(via_settings)
    updated_val_json.update({"_via_img_metadata" :val_dict})
    
    with open(os.path.join(output_dir, "via_region_data_val.json"), 'w') as json_file:
        json.dump(updated_val_json, json_file)
        
if __name__ == '__main__':
    main()