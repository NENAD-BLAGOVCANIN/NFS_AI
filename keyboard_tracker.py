import keyboard

def get_arrow_keys():
    # Initialize the list with all keys set to 0 (not pressed)
    keys_state = [0, 0, 0, 0]  # Order: [left, right, up, down]
    
    # Check each arrow key's state and update the list
    if keyboard.is_pressed("left"):
        keys_state[0] = 1
    if keyboard.is_pressed("right"):
        keys_state[1] = 1
    if keyboard.is_pressed("up"):
        keys_state[2] = 1
    if keyboard.is_pressed("down"):
        keys_state[3] = 1

    return keys_state
