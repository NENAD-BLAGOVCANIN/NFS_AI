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
        
        frames_and_keys.append((frame, keys_pressed))
        frame_count += 1

        if frame_count % batch_size == 0:
            if os.path.exists(output_filename):
                existing_data = np.load(output_filename, allow_pickle=True)
                new_data = np.concatenate((existing_data, np.array(frames_and_keys, dtype=object)))
            else:
                new_data = np.array(frames_and_keys, dtype=object)
            
            np.save(output_filename, new_data, allow_pickle=True)
            frames_and_keys = []

        time.sleep(0.05)

except KeyboardInterrupt:
    print("Recording interrupted by user.")

finally:
    if frames_and_keys:
        if os.path.exists(output_filename):
            existing_data = np.load(output_filename, allow_pickle=True)
            new_data = np.concatenate((existing_data, np.array(frames_and_keys, dtype=object)))
        else:
            new_data = np.array(frames_and_keys, dtype=object)
        np.save(output_filename, new_data, allow_pickle=True)
    
    close_recorder()
    print(f"Frames and keys saved in '{output_filename}'")
