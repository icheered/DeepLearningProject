import pickle
import matplotlib.pyplot as plt

# Load the training history
with open('batch_training_history.pkl', 'rb') as file_pi:
    history = pickle.load(file_pi)

# Plot the training loss and validation loss
plt.plot(history['loss'], label='Training Loss')
plt.title('Model Loss Over Training')
plt.ylabel('Loss')
plt.xlabel('Training step')
plt.legend()
#plt.show()
plt.savefig('training_loss.png')
