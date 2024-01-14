import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.inception_v3 import preprocess_input, InceptionV3

# Instantiate the inception-V3 pre-trained model
model = InceptionV3()
model.load_weights('inception_v3.ckpt')

# print("Done")
# # Assuming you have your NIPS dataset in a directory
# test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

# # Replace 'path_to_nips_dataset' with the path to your NIPS dataset
# test_generator = test_datagen.flow_from_directory(
#     'dataset/images',
#     target_size=(299, 299),
#     batch_size=32,
#     class_mode='categorical')

# # Evaluate the model
# eval_result = model.evaluate(test_generator)
# print(f'Test Loss: {eval_result[0]}, Test Accuracy: {eval_result[1]}')

# # Make predictions
# #predictions = model.predict(test_generator)
# # process predictions as needed
