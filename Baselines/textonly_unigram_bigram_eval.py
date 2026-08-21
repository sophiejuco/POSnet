#import packages
import json
from tqdm import tqdm
import nltk
from nltk import pos_tag
from nltk.tokenize import word_tokenize
import numpy as np
import pandas as pd


#read data - save text-only train, test, val .json files to the same directory as this .py file
train = open("train.json",)
test = open("test.json",)
val = open("val.json",)



def _read_json(file_name):
    """Reads in json lines to create the dataset."""
    all_data = []
    used_recipe_ids = {}

    
    json_file = json.load(file_name)
    data = json_file["data"]

    for data_raw in tqdm(data, total=len(data)):
        recipe_id = data_raw["recipe_id"]
        
        if recipe_id in used_recipe_ids:
            continue

        used_recipe_ids[recipe_id] = True
        context = data_raw["context"]

        rendez_vous = []

        for step in context:
            text = step["body"]
            words = text.split()
            step_id = int(step["id"])
            for word in words:
                rendez_vous.append([word, step_id])

        all_data.extend(rendez_vous)

    return all_data


#generates the true POS tags - saves tokens and tags as lists of sentences/steps which are lists of 
# tuples (token, tag)
def true_tagger(all_data):
    print("Tagging:\n")
    all_sent = []
    word_tags = []
    prev_step = 1
    for entry in tqdm(all_data, total=len(all_data)):
        word, step = entry
        
        current_step = step
        if current_step != prev_step:
            all_sent.append(word_tags)
            word_tags = []
            
        tokens = word_tokenize(word)
        tagged = pos_tag(tokens)

        for token, tag in tagged:
            word_tags.append((token, tag))
        
        prev_step = current_step
        
    all_sent.append(word_tags)

    return all_sent


#read test data & generate true tags
test_all = _read_json(test)
test_tags = true_tagger(test_all)


#read train data & generate true tags
train_all = _read_json(train)
train_tags = true_tagger(train_all)


#read val data & generate true tags
val_all = _read_json(val)
val_tags = true_tagger(val_all)


### Unigram/Bigram Evaluation 


# Loading Libraries
from nltk.tag import UnigramTagger
from nltk.tag import BigramTagger


#initialize/train taggers
uni_tagger = UnigramTagger(train_tags)
bi_tagger = BigramTagger(train_tags)


#sepearate test data in to tokens and tags
X_test = []
y_test = []

for i in range(len(test_tags)):
    X_test.append(list(zip(*test_tags[i]))[0])
    y_test.append(list(zip(*test_tags[i]))[1])


#generate tag predictions for test set on just the test tokens
uni_preds = uni_tagger.tag_sents(X_test)
bi_preds = bi_tagger.tag_sents(X_test)


#separate predicted tags
y_pred_uni = []
y_pred_bi = []

for i in range(len(uni_preds)):
    y_pred_uni.append(list(zip(*uni_preds[i]))[1])
    y_pred_bi.append(list(zip(*bi_preds[i]))[1])
    

#convert to flattened lists
y_test_arr = [item for t in y_test for item in t]
y_pred_uni_arr = [item for t in y_pred_uni for item in t]
y_pred_bi_arr = [item for t in y_pred_bi for item in t]


#expects lists of the same length
#function to generate evaluation metrics
def tag_metrics(actual_arr, predicted_arr):
    
    len_actual = sum(y is not None for y in actual_arr) #don't count None values
    len_pred = sum(y is not None for y in predicted_arr) #don't count None values
    num_correct = 0
    
    for i in range(len(actual_arr)):
        if predicted_arr[i] == actual_arr[i]:
            num_correct += 1
        else:
            continue
            
    #precision
    precision = num_correct/len_pred
    
    #recall
    recall = num_correct/len_actual
    
    #f-measure
    f1 = 2/((1/precision)+(1/recall))
    
    print("precision: " + str(round(precision, 4)))
    print("recall: " + str(round(recall, 4)))
    print("f1: " + str(round(f1, 4)))


#get unigram tagger evaluation
tag_metrics(y_test_arr, y_pred_uni_arr)


#get bigram tagger evaluation
tag_metrics(y_test_arr, y_pred_bi_arr)

