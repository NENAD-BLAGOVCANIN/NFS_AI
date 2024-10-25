from screen_recorder import record_screen, close_recorder
from keyboard_tracker import get_arrow_keys
import numpy as np
import time
import os

# Output file for frames
output_filename = "train_data.npy"
batch_size = 100
frames_and_keys = []

try:
    print("Recording... Press Ctrl+C to stop.")
    frame_count = 0
    
    while True:
        # Capture the screen and get the frame as an array
        frame = record_screen()
        keys_pressed = get_arrow_keys()  # Get the array of keys pressed
        
        # Store the frame and keys pressed as a tuple
        frames_and_keys.append((frame, keys_pressed))
        frame_count += 1

        # Save frames in batches
        if frame_count % batch_size == 0:
            # If the file already exists, append; otherwise, save as a new file
            if os.path.exists(output_filename):
                existing_data = np.load(output_filename, allow_pickle=True)
                new_data = np.concatenate((existing_data, np.array(frames_and_keys, dtype=object)))  # Use dtype=object
            else:
                new_data = np.array(frames_and_keys, dtype=object)  # Use dtype=object
            
            # Save updated data
            np.save(output_filename, new_data, allow_pickle=True)  # Allow pickle for heterogeneous data
            frames_and_keys = []  # Reset frames list after saving

        time.sleep(0.05)

except KeyboardInterrupt:
    print("Recording interrupted by user.")

finally:
    # Save any remaining frames if there are any
    if frames_and_keys:
        if os.path.exists(output_filename):
            existing_data = np.load(output_filename, allow_pickle=True)
            new_data = np.concatenate((existing_data, np.array(frames_and_keys, dtype=object)))  # Use dtype=object
        else:
            new_data = np.array(frames_and_keys, dtype=object)  # Use dtype=object
        np.save(output_filename, new_data, allow_pickle=True)
    
    close_recorder()
    print(f"Frames and keys saved in '{output_filename}'")
