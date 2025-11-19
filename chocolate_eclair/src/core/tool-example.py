#!/usr/bin/env python

# chmod +x *.py

from time import sleep as t_sleep
from spectacle_camera_screenshotter_2 import simple_timer
from spectacle_camera_screenshotter_2 import take_screenshot_flameshot

if __name__ == "__main__":
    output_path = "./screenshots/"
    output_name = "test-flameshot-000"
    output = f"{output_path}{output_name}.jpg"

    # simple_timer()
    take_screenshot_flameshot(output)
