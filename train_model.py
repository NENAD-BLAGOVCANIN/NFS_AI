import os
import numpy as np
from alexnet import alexnet
from sklearn.preprocessing import OneHotEncoder

# Model and Data Parameters
WIDTH = 100
HEIGHT = 80
LR = 1e-4
EPOCHS = 10
MODEL_NAME = 'nfs-driving-{}-{}-{}-epochs-300K-data.model'.format(LR, 'alexnetv2', EPOCHS)

# Update action_map according to the unique actions present in your training data
action_map = {
    0: (0, 0, 0, 0),
    1: (0, 0, 0, 1),
    2: (0, 0, 1, 0),
    3: (0, 1, 0, 1),
    4: (0, 1, 1, 1),
    5: (1, 0, 1, 1),
    6: (1, 1, 0, 0),
    7: (1, 1, 1, 0)
}

# Define model with updated output neurons
model = alexnet(WIDTH, HEIGHT, LR, output=len(action_map))

# Check if a saved model exists
if os.path.exists("model_alexnet-25500" + '.meta'):
    print("Loading previous model...")
    model.load("model_alexnet-25500")
else:
    print("No previous model found. Starting fresh.")

hm_data = 22
for epoch in range(EPOCHS):
    for i in range(1, hm_data + 1):
        # Load training data
        train_data = np.load('training_data.npy', allow_pickle=True)

        # Split data into training and test sets
        train = train_data[:-100]
        test = train_data[-100:]

        # Prepare input data
        X = np.array([data[0] for data in train]).reshape(-1, WIDTH, HEIGHT, 1)

        # Prepare output data
        action_values = [data[1] for data in train]
        Y = np.array(action_values)  # Ensure this is correct as per action_map

        # One-hot encode the actions
        one_hot = OneHotEncoder(sparse_output=False)
        Y_one_hot = one_hot.fit_transform(Y)  # Should be (num_samples, 8)

        print("X shape:", X.shape)  # Should be (num_samples, 100, 80, 1)
        print("Y shape:", Y_one_hot.shape)  # Should be (num_samples, 8)

        # Prepare test data
        test_x = np.array([data[0] for data in test]).reshape(-1, WIDTH, HEIGHT, 1)
        test_y = np.array([data[1] for data in test])
        test_y_one_hot = one_hot.transform(test_y)  # One-hot encode test targets

        # Train the model
        model.fit({'input': X}, {'targets': Y_one_hot}, n_epoch=1,
                  validation_set=({'input': test_x}, {'targets': test_y_one_hot}),
                  snapshot_step=500, show_metric=True, run_id=MODEL_NAME)

        # Save model after each epoch
        model.save(MODEL_NAME)
        print(f"Epoch {epoch + 1}/{EPOCHS} completed and saved.")