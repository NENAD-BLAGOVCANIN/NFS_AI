from screen_recorder import record_screen, close_recorder
from keyboard_tracker import get_arrow_keys
import numpy as np
import time
import os

output_filename = "train_data.npy"
batch_size = 100
frames_and_keys = []

try:
    print("Recording... Press Ctrl+C to stop.")
    frame_count = 0
    
    while True:

        frame = record_screen()
        keys_pressed = get_arrow_keys()
        
        # Ensure frames and keys have a consistent structure
        if frame is None or frame.ndim != 2:
            raise ValueError("Captured frame must be a 2D grayscale array")
        if not isinstance(keys_pressed, (list, tuple)):
            raise ValueError("Keys pressed should be a list or tuple")
        
        frames_and_keys.append((frame, keys_pressed))
        frame_count += 1

        if frame_count % batch_size == 0:
            # Convert to array with consistent shape
            batch_data = np.array(frames_and_keys, dtype=object)
            if os.path.exists(output_filename):
                existing_data = np.load(output_filename, allow_pickle=True)
                new_data = np.concatenate((existing_data, batch_data))
            else:
                new_data = batch_data
            
            np.save(output_filename, new_data, allow_pickle=True)
            frames_and_keys = []

        time.sleep(0.05)

except KeyboardInterrupt:
    print("Recording interrupted by user.")

finally:
    if frames_and_keys:
        batch_data = np.array(frames_and_keys, dtype=object)
        if os.path.exists(output_filename):
            existing_data = np.load(output_filename, allow_pickle=True)
            new_data = np.concatenate((existing_data, batch_data))
        else:
            new_data = batch_data
        np.save(output_filename, new_data, allow_pickle=True)
    
    close_recorder()
    print(f"Frames and keys saved in '{output_filename}'")
