import pygame
def play_sound( mp3_file):
    pygame.mixer.init()
    pygame.mixer.music.load(mp3_file)
    pygame.mixer.music.play()
    is_sound_playing = True
    while pygame.mixer.music.get_busy():
        pygame.time.Clock().tick(10)
    is_sound_playing = False

mp3_file = '/home/kart/yolo_test/sounds/ready.mp3' #ready
play_sound(mp3_file)