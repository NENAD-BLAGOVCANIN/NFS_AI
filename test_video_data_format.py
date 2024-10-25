import numpy as np
import cv2

def load_numpy_file(file_path):
    """Load the NumPy array from the specified file."""
    return np.load(file_path)

def create_video_from_grayscale_array(gray_array, output_file, fps=30):
    """Create a video from the given grayscale array."""
    # Get the shape of the array
    height, width = gray_array.shape[1], gray_array.shape[2]
    
    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'XVID')  # You can change codec if needed
    out = cv2.VideoWriter(output_file, fourcc, fps, (width, height), False)

    for frame in gray_array:
        # Convert frame to uint8 if necessary
        if frame.dtype != np.uint8:
            frame = (frame * 255).astype(np.uint8)
        out.write(frame)

    out.release()
    print(f"Video saved to {output_file}")

if __name__ == "__main__":

    numpy_file_path = 'train_data.npy'
    output_video_path = 'test_video.avi'

    gray_array = load_numpy_file(numpy_file_path)

    # Ensure the array is in the correct shape (number of frames, height, width)
    if gray_array.ndim != 3:
        raise ValueError("The input array must have shape (frames, height, width).")

    create_video_from_grayscale_array(gray_array, output_video_path)