from sklearn.model_selection import train_test_split
import numpy as np
import os
import csv
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten, Input, Concatenate
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import Embedding  # Import the Embedding layer
from tensorflow.keras.layers import GlobalAveragePooling2D  # Import the GlobalAveragePooling2D layer
#from RecipeQA_dataset import _read_image_paths, _read_json, true_tagger
import tensorflow as tf
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.metrics import Precision, Recall


class BatchEndCallback(Callback):
    def on_batch_end(self, batch, logs=None):
        logs = logs or {}
        print(f"End of batch {batch}, Loss: {logs.get('loss')}, Accuracy: {logs.get('accuracy')}")

class EpochEndCallback(Callback):
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        print(f"End of epoch {epoch}, Loss: {logs.get('loss')}, Accuracy: {logs.get('accuracy')}")


print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))


print('Starting...')

def read_data_from_csv(file_name, limit=1000000):
    words = []
    pos_tags = []
    image_paths = []

    with open(file_name, mode='r', newline='', encoding='utf-8') as file:
        reader = csv.reader(file)
        next(reader)  # Skip the header row
        for i, row in enumerate(reader):
            if i >= limit:  # Stop after reading limit rows
                break
            words.append(row[0])
            pos_tags.append(row[1])
            image_paths.append(row[2])

    return words, pos_tags, image_paths

csv_file_name = '/scratch/baj321/RecipeCode/RecipeQA_dataset.csv'  # Path to your CSV file
words, pos_tags, image_paths = read_data_from_csv(csv_file_name)

print('Data Loaded')

# First, split the data into a training set and a temporary set (combining test and validation)
train_words, temp_words, train_tags, temp_tags, train_images, temp_images = train_test_split(
    words, pos_tags, image_paths, test_size=0.3,train_size=0.7, random_state=42)

# Now, split the temporary set into test and validation sets
test_words, val_words, test_tags, val_tags, test_images, val_images = train_test_split(
    temp_words, temp_tags, temp_images, test_size=0.5, random_state=42)


print("train words:",len(train_words))
print("train tags:",len(train_tags))
print("train images:",len(train_images))

print("val words:",len(val_words))
print("val tags:",len(val_tags))
print("val images:",len(val_images))

# Convert words to sequences
tokenizer = Tokenizer()
tokenizer.fit_on_texts(train_words)
sequences = tokenizer.texts_to_sequences(train_words)
word_index = tokenizer.word_index

# Padding sequences to ensure uniform input size
max_length = max([len(seq) for seq in sequences])
word_seq = pad_sequences(sequences, maxlen=max_length)


# Process images with resizing
target_size = (32, 32)  # Example size, you can adjust this

# Process images
def load_and_preprocess_image(img_path):
    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = preprocess_input(img_array)
    return img_array

image_data = np.stack([load_and_preprocess_image(path) for path in train_images])

# Convert POS tags to categorical labels
unique_tags = list(set(train_tags))
tag_index = dict((tag, i) for i, tag in enumerate(unique_tags))
labels = to_categorical([tag_index[tag] for tag in train_tags])
# print("Labels:",labels)

# Text Model
text_input = Input(shape=(max_length,))
text_layer = Embedding(len(word_index) + 1, 100, input_length=max_length)(text_input)
text_layer = Flatten()(text_layer)

# Image Model
image_input = Input(shape=target_size + (3,))
base_model = VGG16(include_top=False, weights='imagenet', input_shape=target_size + (3,))
base_model.trainable = False
image_layer = base_model(image_input)
image_layer = GlobalAveragePooling2D()(image_layer)

# Concatenate both models
combined = Concatenate()([text_layer, image_layer])

# Add fully connected layers
combined = Dense(256, activation='relu')(combined)
combined = Dense(len(unique_tags), activation='softmax')(combined)

# Final model
model = Model(inputs=[text_input, image_input], outputs=combined)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy', Precision(), Recall()])


class F1ScoreCallback(Callback):
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        precision = logs.get('precision')
        recall = logs.get('recall')
        if precision is not None and recall is not None and precision + recall > 0:
            f1_score = 2 * (precision * recall) / (precision + recall)
            logs['f1_score'] = f1_score
            print(f" — val_f1: {f1_score:.4f}")
        else:
            logs['f1_score'] = 0
            print(" — val_f1: N/A (precision or recall is zero)")


# Process Test Set
test_sequences = tokenizer.texts_to_sequences(test_words)
test_word_seq = pad_sequences(test_sequences, maxlen=max_length)
test_image_data = np.stack([load_and_preprocess_image(path) for path in test_images])
test_labels = to_categorical([tag_index.get(tag, 0) for tag in test_tags])  # using 'get' to handle unseen tags

# Process Validation Set
val_sequences = tokenizer.texts_to_sequences(val_words)
val_word_seq = pad_sequences(val_sequences, maxlen=max_length)
val_image_data = np.stack([load_and_preprocess_image(path) for path in val_images])
val_labels = to_categorical([tag_index.get(tag, 0) for tag in val_tags])


print("Words:", len(words), "POS Tags:", len(pos_tags), "Image Paths:", len(image_paths))
print("Train Words:", len(train_words), "Train POS Tags:", len(train_tags), "Train Image Paths:", len(train_images))
print("Test Words:", len(test_words), "Test POS Tags:", len(test_tags), "Test Image Paths:", len(test_images))
print("Val Words:", len(val_words), "Train POS Tags:", len(val_tags), "Train Image Paths:", len(val_images))
print(np.array(word_seq).shape, np.array(image_data).shape, np.array(labels).shape)
# Train the model with validation data
model.fit([word_seq, image_data], labels, validation_data=([val_word_seq, val_image_data], val_labels), epochs=100, batch_size=4, verbose=1, callbacks=[ F1ScoreCallback()])


# Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate([test_word_seq, test_image_data], test_labels)
print("Test Accuracy:", test_accuracy)

# Note: Ensure that the paths in 'test_images' and 'val_images' are correct and accessible.