import json
import os
from sklearn.model_selection import train_test_split
import numpy as np
from tensorflow.keras.preprocessing import image  # Correct import
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.models import Model
import glob
from tqdm import tqdm
import nltk
from nltk import pos_tag
from nltk.tokenize import word_tokenize
import csv
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

def _read_image_paths( data_dir, split):
        img_paths_dict = {}
        
        img_dir = os.path.join(data_dir, "images", "images-qa",
                               split, "images-qa")
        all_img_paths = glob.glob(os.path.join(img_dir, "*.jpg"))
        for img_path_raw in sorted(all_img_paths):
            img_path = img_path_raw.strip()
            img_name = img_path.split("/")[-1].split(".")[0]

            img_name_splits = img_name.split("_")
            if not img_name_splits[-2].isdigit():
                recipe_id = "_".join(img_name.split("_")[:-1])
                step_id = int(img_name.split("_")[-1])
            else:
                recipe_id = "_".join(img_name.split("_")[:-2])
                step_id = int(img_name.split("_")[-2])
                img_id = int(img_name.split("_")[-1])
            if recipe_id not in img_paths_dict:
                img_paths_dict[recipe_id] = {}
            if step_id not in img_paths_dict[recipe_id]:
                img_paths_dict[recipe_id][step_id] = []
            #print(recipe_id)
            img_paths_dict[recipe_id][step_id].append(img_path_raw)
        return img_paths_dict



def _read_json(data_dir):
    """Reads in json lines to create the dataset."""
    all_data = []
    for name in ["test", "val"]:    
        json_path = os.path.join(data_dir, "texts", name + ".json") 
        image_paths = _read_image_paths(data_dir=data_dir,split= name)
        
        #print(image_paths)
        used_recipe_ids = {}

        
        json_file = json.load(open(json_path))
        data = json_file["data"]

        for data_raw in tqdm(data, total=len(data)):
            recipe_id = data_raw["recipe_id"]
            
            if recipe_id in used_recipe_ids:
                continue

            used_recipe_ids[recipe_id] = True
            context = data_raw["context"]
            file_path = os.path.join(data_dir,"images", "images-qa",name, "images-qa", recipe_id)
            #print(file_path)

            image_paths_curr = image_paths.get(recipe_id, {})
            #print(image_paths_curr)
            rendez_vous = []

            for step in context:
                text = step["body"]
                words = text.split()
                step_id = int(step["id"])
                for word in words:
                    image_list = image_paths_curr.get(step_id, {})
                    if image_list:
                        for image in image_list:
                            rendez_vous.append([word, step_id, image])

            all_data.extend(rendez_vous)

    return all_data


def true_tagger(all_data):
    print("Tagging:\n")
    word_image_tags = []
    for entry in tqdm(all_data, total=len(all_data)):
        word, step, image = entry
        tokens = word_tokenize(word)
        tagged = pos_tag(tokens)

        for token, tag in tagged:
            word_image_tags.append([token, tag, image])

    return word_image_tags
    
def write_to_csv(data, file_name):
    """
    Writes the given data to a CSV file.

    :param data: List of lists, where each sublist contains data for one row.
    :param file_name: Name of the CSV file to write to.
    """
    with open(file_name, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        # Write the header
        writer.writerow(['Token', 'Tag', 'Image'])

        # Write the data
        for row in data:
            writer.writerow(row)
    
    
# Your existing code to process the data
data_dir = '/scratch/baj321/RecipeQA/'  # Replace with your actual data directory
all_data = _read_json(data_dir)
word_image_tags = true_tagger(all_data)

# Write to CSV
csv_file_name = 'RecipeQA_dataset.csv'  # Name of your output file
write_to_csv(word_image_tags, csv_file_name)    

# RECIPEQA_DATA_ROOT = "C:/Users/Baraa/Desktop/recipeqa_acl22_data-002/"
# word_paths_paired = _read_json(data_dir=RECIPEQA_DATA_ROOT)
# tagged = true_tagger(word_paths_paired)
# print(tagged)