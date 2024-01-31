import pickle
import matplotlib.pyplot as plt

# Load the training history
with open('batch_training_history.pkl', 'rb') as file_pi:
    history = pickle.load(file_pi)

# Plot the training loss and validation loss
plt.plot(history['loss'], label='Training Loss')
plt.plot(history['val_loss'], label='Validation Loss')
plt.title('Model Loss Over Epochs')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend()
plt.show()
