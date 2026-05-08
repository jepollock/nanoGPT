
enabled = False

def set_debug_flag(boolean):
    global enabled
    enabled = boolean

def debug(string):
    if (enabled):
        print("JASON: " + string)
