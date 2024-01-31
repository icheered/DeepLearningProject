import random

import numpy as np
import pandas as pd
import numpy as np

from tqdm import tqdm
import pickle

from sklearn.model_selection import train_test_split

from keras import Model
from keras.preprocessing.image import load_img, img_to_array

from keras.callbacks import ModelCheckpoint
from keras.layers import Input, Conv2D, Conv2DTranspose, MaxPooling2D, ZeroPadding2D, Flatten, Dense, Reshape


def load_and_preprocess_image(image_path, target_size=(299, 299)):
    image = load_img(image_path, target_size=target_size)
    image = img_to_array(image)
    image = image / 255.0  # Normalize to [0, 1]
    return image

# Data generator
def data_generator(clean_paths, poisoned_paths, batch_size):
    while True:
        for i in range(0, len(clean_paths), batch_size):
            batch_clean = [load_and_preprocess_image(clean_paths[i]) for i in range(i, min(i + batch_size, len(clean_paths)))]
            batch_poisoned = [load_and_preprocess_image(poisoned_paths[i]) for i in range(i, min(i + batch_size, len(poisoned_paths)))]

            yield np.array(batch_poisoned), np.array(batch_clean)



def build_full_model(input_layer):
    # Input layer
    inputs = (input_layer)  # Assuming RGB images

    # Encoder
    conv1 = Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    conv1 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool1)
    conv2 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool2)
    conv3 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    # Bottom
    conv4 = Conv2D(512, (3, 3), activation='relu', padding='same')(pool3)
    conv4 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv4)

    # Decoder
    up5 = Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(conv4)
    conv5 = Conv2D(256, (3, 3), activation='relu', padding='same')(up5)
    conv5 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv5)

    up6 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv5)
    conv6 = Conv2D(128, (3, 3), activation='relu', padding='same')(up6)
    conv6 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv6)

    up7 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv6)
    conv7 = Conv2D(64, (3, 3), activation='relu', padding='same')(up7)
    conv7 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv7)

    # Zero-padding layer to adjust the dimensions
    padded = ZeroPadding2D(padding=((2, 1), (2, 1)))(conv7)

    # Output layer
    decoded = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(padded)

    return decoded

def build_model(input_layer):
    # Input layer
    inputs = input_layer  # Assuming RGB images

    # Encoder
    conv1 = Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    conv2 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv1)
    conv3 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv2)

    # Decoder
    conv4 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv3)
    conv5 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv4)

    # Output layer (no need for zero padding if the spatial dimensions haven't been altered)
    decoded = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(conv5)

    return decoded

def build_shallow_model(input_shape):
    # Input layer
    inputs = Input(shape=input_shape)  # Assuming RGB images (299, 299, 3)
    
    # Flatten the input
    flat_input = Flatten()(inputs)

    # Hidden layer
    encoded = Dense(units=512, activation='relu')(flat_input)

    # Output layer
    flat_output = Dense(units=299*299*3, activation='sigmoid')(encoded)
    
    # Reshape back to the original image size
    decoded = Reshape(target_shape=input_shape)(flat_output)

    return Model(inputs, decoded)

# Create the model
input_image = Input(shape=(299, 299, 3))
model = Model(input_image, build_model(input_image))
model.compile(optimizer='adam', loss='MSE')
model.summary()


# input_image=Input(shape=(299, 299, 3))
# decoded=build_model(input_image)
# model = Model(input_image, decoded)
# model.compile(optimizer='adam', loss='MSE')
# model.summary()

# Create the model
# input_shape = (299, 299, 3)
# model = build_shallow_model(input_shape)
# model.compile(optimizer='adam', loss='MSE')
# model.summary()

# Read the CSV
data_csv = pd.read_csv('poisoned_data.csv') 

# Locations of the images
clean_images_dir = "dataset/images"
poisoned_images_dir = "poisoned_data"

# Prepare pairs of file paths
image_pairs = [(f"{clean_images_dir}/{row['original_filename']}", f"{poisoned_images_dir}/{row['poisoned_filename']}") for _, row in data_csv.iterrows()]

# Make a random selection to test the training process. Shuffle the list first.
random.shuffle(image_pairs)
image_pairs = image_pairs[:100]

# Split the dataset into training and validation sets
train_pairs, val_pairs = train_test_split(image_pairs, test_size=0.2, random_state=42)
train_clean_paths, train_poisoned_paths = zip(*train_pairs)
val_clean_paths, val_poisoned_paths = zip(*val_pairs)

print(f"Training samples: {len(train_clean_paths)}, Validation samples: {len(val_clean_paths)}")

# Create data generators
batch_size = 8
train_generator = data_generator(train_clean_paths, train_poisoned_paths, batch_size)
val_generator = data_generator(val_clean_paths, val_poisoned_paths, batch_size)

# Calculate steps per epoch and validation steps
steps_per_epoch = len(train_clean_paths) // batch_size
validation_steps = len(val_clean_paths) // batch_size
print(f"Steps per epoch: {steps_per_epoch}, Validation steps: {validation_steps}")

try:
    # Training the model
    history = model.fit(train_generator,
                        steps_per_epoch=steps_per_epoch,
                        validation_data=val_generator,
                        validation_steps=validation_steps,
                        epochs=2) 
except KeyboardInterrupt:
    print("\nTraining interrupted by user.")
finally:
    # Save the model
    print("Saving model")
    model.save("denoiser_model.keras")
    print("Model saved.")
    
    # Save the history
    print("Saving training history")
    with open('training_history.pkl', 'wb') as file_pi:
        pickle.dump(history.history, file_pi)
    print("Training history saved.")