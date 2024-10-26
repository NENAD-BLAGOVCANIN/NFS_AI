import cv2
import numpy as np
import mss

# Screen and capture settings
SCREEN_WIDTH = 2880
SCREEN_HEIGHT = 1620
CAPTURE_AREA_WIDTH = 1100
CAPTURE_AREA_HEIGHT = 900

x = SCREEN_WIDTH // 2 - (CAPTURE_AREA_WIDTH // 2)
y = SCREEN_HEIGHT // 2 - (CAPTURE_AREA_HEIGHT // 2) + 300

# Initialize screen capture with mss
sct = mss.mss()
monitor = {"top": y, "left": x, "width": CAPTURE_AREA_WIDTH, "height": CAPTURE_AREA_HEIGHT}

def record_screen():
    sct_img = sct.grab(monitor)
    frame = np.array(sct_img)
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2GRAY)

    # cv2.imshow("Screen Capture", gray_frame)
    
    # # Allow 'q' to quit from the display window
    # if cv2.waitKey(1) & 0xFF == ord('q'):
    #     close_recorder()
    #     raise SystemExit  # Stop the external loop if 'q' is pressed

    return gray_frame

def close_recorder():
    sct.close()
    cv2.destroyAllWindows()
    print("Screen capture session closed.")
