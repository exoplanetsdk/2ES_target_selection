import subprocess
import threading
import time
import pygame
import ipywidgets as widgets
from IPython.display import display

# Global flag to control the speaking loop
stop_flag = False

def speak_text_continuously(text, voice='Ava'):
    global stop_flag
    while not stop_flag:
        subprocess.run(['say', '-v', voice, text])

def start_speaking(text, voice='Ava'):
    # Start speaking in a separate thread
    thread = threading.Thread(target=speak_text_continuously, args=(text, voice))
    thread.start()
    return thread

def stop_speaking():
    global stop_flag
    stop_flag = True

def play_local_audio(file_path):
    # Initialize pygame mixer
    pygame.mixer.init()
    pygame.mixer.music.load(file_path)
    pygame.mixer.music.play()

def stop_music():
    # Stop the music playback
    if pygame.mixer.music.get_busy():
        pygame.mixer.music.stop()

# Function to start the audio process
def start_audio_process():
    # Start speaking the text
    start_speaking("Hello, the code has finished running.", voice='Ava')
    time.sleep(2.5)  # Adjust this delay as needed
    stop_speaking()
    play_local_audio('../data/inspirational-uplifting-calm-piano.mp3')

# Create a button for stopping music
stop_button = widgets.Button(description="Stop Music")

# Define button actions
def on_stop_button_clicked(b):
    stop_music()

# Attach actions to the stop button
stop_button.on_click(on_stop_button_clicked)