#!/usr/bin/env python

# chmod +x *.py

import subprocess as s
from os.path import exists
from time import sleep as t_sleep

SFX_TIMER_WAIT = "./sounds/completion-partial.wav"
SFX_TIMER_GO = "./sounds/completion-success.wav"
SFX_SCREENSHOT = "./sounds/completion-rotation.wav"
SFX_UNUSED = "./sounds/completion-fail.wav"
SFX_END = "./sounds/message-new-instant.wav"

stop_flag = False
play_sfx = True

def play(sfx:str) -> None:
    if not play_sfx or not exists(sfx):
        return

    s.run(f"aplay {sfx}", shell=True, stdout=s.DEVNULL, stderr=s.DEVNULL)
    return

def simple_timer(seconds: int = 5) -> bool:
    global stop_flag

    try:
        for y in range(1, seconds):
            if stop_flag:
                print("Timer interrupted.")
                return False
            print(f"{seconds - y}…")
            play(SFX_TIMER_WAIT)
            t_sleep(1)

        print("Go!")
        play(SFX_TIMER_GO)

    except KeyboardInterrupt:
        stop_flag = True
        return False

    return True

# def take_screenshot_spectacle(output_file: str) -> None:
#     command = [
#         "spectacle",
#         "--fullscreen",
#         "--background",
#         "--output", output_file
#     ]

#     try:
#         s.run(command, check=True, stdout=s.DEVNULL, stderr=s.DEVNULL)
#         print(f"Screenshot saved to: {output_file}")
#         play(SFX_SCREENSHOT)
#     except s.CalledProcessError as e:
#         print(f"Error capturing screenshot: {e}")
#     return

def take_screenshot_flameshot(output_file: str) -> None:
    command = [
        "flameshot",
        "full",
        "--path", output_file
    ]

    try:
        s.run(command, check=True, stdout=s.DEVNULL, stderr=s.DEVNULL)
        print(f"Screenshot saved to: {output_file}")
        play(SFX_SCREENSHOT)
    except s.CalledProcessError as e:
        print(f"Error capturing screenshot: {e}")
    return

def take_multiple_screenshots(output_dir: str, amount: int, delay) -> None:
    global stop_flag
    for y_not in range(amount):
        if stop_flag:
            print("Screenshot capturing interrupted.")
            break
        path = f"{output_dir}flameshot_{y_not:03}.jpg"
        # play(SFX_SCREENSHOT)
        take_screenshot_flameshot(path)
        t_sleep(delay)

if __name__ == "__main__":
    output_path = "./screenshots/"
    try:
        if simple_timer(seconds=3):
            # run screenshots only if the timer completes without interruption
            take_multiple_screenshots(output_path, amount=15, delay=0.2)
            play(SFX_END)

    except KeyboardInterrupt:
        print("Ctrl+C pressed. Exiting...")
        stop_flag = True
