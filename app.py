from screen_recorder import record_screen, close_recorder
from keyboard_tracker import get_arrow_keys
import numpy as np
import time
import os

# Output file for frames
output_filename = "train_data.npy"
batch_size = 100
frames = []

try:
    print("Recording... Press Ctrl+C to stop.")
    frame_count = 0
    
    while True:
        # Capture the screen and get the frame as an array
        frame = record_screen()
        frames.append(frame)
        frame_count += 1

        # Check arrow keys (optional, for your display purpose)
        print(get_arrow_keys())
        
        # Save frames in batches
        if frame_count % batch_size == 0:
            # If the file already exists, append; otherwise, save as a new file
            if os.path.exists(output_filename):
                existing_data = np.load(output_filename)
                new_data = np.concatenate((existing_data, np.array(frames)))
            else:
                new_data = np.array(frames)
            
            # Save updated data
            np.save(output_filename, new_data)
            frames = []  # Reset frames list after saving

        time.sleep(0.05)

except KeyboardInterrupt:
    print("Recording interrupted by user.")

finally:
    # Save any remaining frames if there are any
    if frames:
        if os.path.exists(output_filename):
            existing_data = np.load(output_filename)
            new_data = np.concatenate((existing_data, np.array(frames)))
        else:
            new_data = np.array(frames)
        np.save(output_filename, new_data)
    
    close_recorder()
    print(f"Frames saved in '{output_filename}'")
