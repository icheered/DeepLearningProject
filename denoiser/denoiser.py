import random

import numpy as np
import pandas as pd
import numpy as np

from tqdm import tqdm
import pickle

from sklearn.model_selection import train_test_split

from keras import Model
from keras.preprocessing.image import load_img, img_to_array
from keras.callbacks import Callback

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

def build_model_small(input_layer):
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

def build_model(input_layer):
    # Input layer
    inputs = input_layer  # Assuming RGB images

    # Encoder
    conv1 = Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    conv2 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv1)
    conv3 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv2)
    conv3a = Conv2D(256, (3, 3), activation='relu', padding='same')(conv3)  # Added extra layer
    conv3b = Conv2D(256, (3, 3), activation='relu', padding='same')(conv3a) # Added extra layer

    # Decoder
    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv3b)  # Increased depth
    conv4a = Conv2D(256, (3, 3), activation='relu', padding='same')(conv4)  # Added extra layer
    conv5 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv4a)
    conv5a = Conv2D(128, (3, 3), activation='relu', padding='same')(conv5)  # Added extra layer
    conv6 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv5a)

    # Output layer (no need for zero padding if the spatial dimensions haven't been altered)
    decoded = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(conv6)

    return decoded

# Create the model
input_image = Input(shape=(299, 299, 3))
model = Model(input_image, build_model_small(input_image))
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
data_csv = pd.read_csv('../poisoned_data.csv') 

# Locations of the images
clean_images_dir = "../dataset/images"
poisoned_images_dir = "../poisoned_data"

# Prepare pairs of file paths
image_pairs = [(f"{clean_images_dir}/{row['original_filename']}", f"{poisoned_images_dir}/{row['poisoned_filename']}") for _, row in data_csv.iterrows()]

# Make a random selection to test the training process. Shuffle the list first.
random.shuffle(image_pairs)

# Split the dataset into training and validation sets
train_pairs, val_pairs = train_test_split(image_pairs, test_size=0.2, random_state=42)
train_clean_paths, train_poisoned_paths = zip(*train_pairs)
val_clean_paths, val_poisoned_paths = zip(*val_pairs)

print(f"Training samples: {len(train_clean_paths)}, Validation samples: {len(val_clean_paths)}")

# Create data generators
batch_size = 16
epochs = 2
train_generator = data_generator(train_clean_paths, train_poisoned_paths, batch_size)
val_generator = data_generator(val_clean_paths, val_poisoned_paths, batch_size)

# Calculate steps per epoch and validation steps
steps_per_epoch = len(train_clean_paths) // batch_size
validation_steps = len(val_clean_paths) // batch_size
print(f"Steps per epoch: {steps_per_epoch}, Validation steps: {validation_steps}")


class BatchHistory(Callback):
    def on_train_begin(self, logs=None):
        self.history = {'loss': [], 'val_loss': []}

    def on_batch_end(self, batch, logs=None):
        # Update the history dictionary with the loss and accuracy for the batch
        for k, v in logs.items():
            if k in self.history:
                self.history[k].append(v)

    def on_epoch_end(self, epoch, logs=None):
        # Save the history after each epoch
        with open('batch_training_history.pkl', 'wb') as file_pi:
            pickle.dump(self.history, file_pi)

# Create an instance of the custom callback
batch_history = BatchHistory()

# Create the ModelCheckpoint callback to save the model after each epoch
model_checkpoint = ModelCheckpoint(
    'denoiser_model_epoch{epoch:02d}.keras', 
    save_best_only=False, 
    verbose=1
)

try:
    # Training the model
    model.fit(
        train_generator,
        steps_per_epoch=steps_per_epoch,
        validation_data=val_generator,
        validation_steps=validation_steps,
        epochs=epochs,
        callbacks=[batch_history, model_checkpoint]
    )
except KeyboardInterrupt:
    print("\nTraining interrupted by user.")
finally:
    # Save the model
    print("Saving model")
    model.save("denoiser_model.keras")
    print("Model saved.")
