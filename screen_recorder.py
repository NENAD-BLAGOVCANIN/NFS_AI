import cv2
import numpy as np
import mss

# Screen and capture settings
SCREEN_WIDTH = 2880
SCREEN_HEIGHT = 1620
CAPTURE_AREA_WIDTH = 1000
CAPTURE_AREA_HEIGHT = 800

x = SCREEN_WIDTH // 2 - (CAPTURE_AREA_WIDTH // 2)
y = SCREEN_HEIGHT // 2 - (CAPTURE_AREA_HEIGHT // 2) + 400

# Initialize screen capture with mss
sct = mss.mss()
monitor = {"top": y, "left": x, "width": CAPTURE_AREA_WIDTH, "height": CAPTURE_AREA_HEIGHT}

def record_screen():
    sct_img = sct.grab(monitor)
    frame = np.array(sct_img)
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2GRAY)

    resized_frame = cv2.resize(gray_frame, (100, 80))

    # cv2.imshow("Resized Frame", gray_frame)

    return resized_frame

def close_recorder():
    sct.close()
    cv2.destroyAllWindows()
    print("Screen capture session closed.")
