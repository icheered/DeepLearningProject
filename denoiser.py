
import numpy as np
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split

from keras import Model
from keras.preprocessing.image import load_img, img_to_array

from keras.callbacks import ModelCheckpoint
from keras.layers import Input, Conv2D, Conv2DTranspose, MaxPooling2D, ZeroPadding2D




def load_and_preprocess_image(image_path, target_size=(299, 299)):
    image = load_img(image_path, target_size=target_size)
    image = img_to_array(image)
    image = image / 255.0  # Normalize to [0, 1]
    return image

def load_dataset(data_csv, clean_dir, poisoned_dir):
    clean_images = []
    poisoned_images = []

    for index, row in data_csv.iterrows():
        clean_path = f"{clean_dir}/{row['original_filename']}"
        poisoned_path = f"{poisoned_dir}/{row['poisoned_filename']}"

        clean_img = load_and_preprocess_image(clean_path)
        poisoned_img = load_and_preprocess_image(poisoned_path)

        clean_images.append(clean_img)
        poisoned_images.append(poisoned_img)

    return np.array(clean_images), np.array(poisoned_images)


def data_generator(clean_images, poisoned_images, batch_size):
    while True:
        for i in range(0, len(clean_images), batch_size):
            yield poisoned_images[i:i+batch_size], clean_images[i:i+batch_size]


def build_model(input_layer):
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



# Create the model
input_image=Input(shape=(299, 299, 3))
decoded=build_model(input_image)

model = Model(input_image, decoded)
model.compile(optimizer='adam', loss='MSE')
model.summary()

#exit()



# First column contains original clean filenames, and second column contains poisoned filenames
data_csv = pd.read_csv('poisoned_data.csv') 

# Locations of the images
clean_images_dir = "dataset/images"
poisoned_images_dir = "poisoned_data"

# Load the dataset
print("Loading dataset")
clean_imgs, poisoned_imgs = load_dataset(data_csv, clean_images_dir, poisoned_images_dir)

# Split the dataset into training and validation sets
print("Splitting dataset")
train_poisoned, val_poisoned, train_clean, val_clean = train_test_split(poisoned_imgs, clean_imgs, test_size=0.2, random_state=42)

# Create data generators
print("Creating data generators")
batch_size = 4  # Adjust based on your GPU capacity
train_generator = data_generator(train_clean, train_poisoned, batch_size)
val_generator = data_generator(val_clean, val_poisoned, batch_size)

steps_per_epoch = len(train_clean) // batch_size
validation_steps = len(val_clean) // batch_size

print(f"Steps per epoch: {steps_per_epoch}, validation steps: {validation_steps}")

print("Training model")
# Train the model
model.fit(train_generator,
          steps_per_epoch=len(train_clean) // batch_size,
          validation_data=val_generator,
          validation_steps=len(val_clean) // batch_size,
          epochs=1)  

# Save the model
print("Saving model")
model.save("denoiser_model.keras")
