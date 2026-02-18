from ultralytics import YOLO
from short import Sort
import math
import time
import cv2
import threading
import serial
import pygame
import numpy as np

TURN_MIN_VALUE = 75

DISTANCE_MIN_VALUE = 10

# 시리얼 포트 설정
serial_port = '/dev/ttyACM1'  # 실제 포트에 맞게 수정해야 함
baudrate = 9600 # 이거 아두이노에 설정한 값이랑 같아야 한다.

# 시리얼 연결 설정
ser = serial.Serial(serial_port, baudrate) # 시리얼로 연결

is_sound_playing = False
last_sound_time = 0

pTime = 0
sound_time_2min = time.time()
frame_count = 0

# 명령 전송
def send_command(char, num1=0): # send_command는 블루투스를 통해 명령을 아두이노로 전송하는 함수.\
    # num의 기본 값은 0으로 설정해서 정지할 때는 pwm 값을 아두이노로 안 보내도 된다.
    signal = f"{char},{num1}\n" # 문자와 숫자를 쉼표로 구분해서 전송. 
    ser.write(signal.encode()) # command 문자열을 바이트 형식으로 변환하고 ser을 통해 전송.
    print(f"명령 '{signal}' 전송됨")

def RangeCalc(In, in_max, in_min, out_max, out_min):
    x = min(max(In, in_min), in_max)
    return (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min

# def play_sound(mp3_file):
#     pygame.mixer.init()
#     pygame.mixer.music.load(mp3_file)
#     pygame.mixer.music.play()
#     is_sound_playing = True
#     while pygame.mixer.music.get_busy(): #play 중이면
#         pygame.time.Clock().tick(10) #루프가 10초에 한 번씩 돌게 함
#     is_sound_playing = False

# 음성 재생동안 코드가 진행되지 않게 대기하는 함수        
# def wait_with_time(duration):
#    start_time = time.time()
#    while time.time() - start_time < duration:
#        time.sleep(0.01)

# Initialize the webcam
cap = cv2.VideoCapture(0)
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

tracker = Sort(max_age=50, min_hits=3, iou_threshold=0.1)
# Initialize yolov8 object detector
model_path = "/home/kart/yolo_test/best_full_integer_quant_edgetpu.tflite"
#model_path = "/home/kart/yolo_test/240_yolov8n_full_integer_quant_edgetpu.tflite"
model = YOLO(model_path)

mp3_file = '/home/soeunan/Kartriders/sounds/intro.mp3'
#play_sound(mp3_file)


mp3_file = '/home/soeunan/Kartriders/sounds/ready.mp3'
#play_sound(mp3_file)

while cap.isOpened():

    # 프레임 계산
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    
    # Read frame from the video
    ret, frame = cap.read()
    if not ret:
        break

    # 캠 프레임의 이미지 크기와 좌표
    frame_h, frame_w, _ = frame.shape
    frame_center_x = frame_w // 2
    frame_center_y = frame_h // 2
    frame_center = [frame_center_x, frame_center_y]
    frame_center_tu = (int(frame_center_x), int(frame_center_y))

    # 객체 탐지
    # boxes, scores, class_ids = yolov8_detector(frame)
    results = model.predict(source=frame, conf=0.7, iou=0.3, save=False, verbose=False)
    results= results[0]
    class_ids = results.boxes.cls.cpu().numpy()
    boxes = results.boxes.xyxy.cpu().numpy()

    if 1 in class_ids:
        print("휠체어 사용자가 탐지됨")
        # 바운딩 박스 정보를 얻음
        for box, class_id in zip(boxes, class_ids):
            if class_id == 1:
                tracks = tracker.update(boxes)
                tracks = tracks.astype(int)
                #print(tracks[-1])
                if len(tracks) > 0:
                    xmin, ymin, xmax, ymax, track_id = tracks[-1].astype(int)
                    box_h = ymax - ymin
                    box_w = xmax - xmin
                    box_center_x = xmin + (box_w // 2)
                    box_center_y = ymin + (box_h // 2)
                    box_center = [box_center_x, box_center_y]
                    box_center_tu = (box_center_x, box_center_y)

                    turn_direc = frame_center_x - box_center[0]
                    distance_scale = frame_h - ymax

                    if frame_count % 50 == 0:

                        # 거리 제어
                        if distance_scale <= 19:
                            distance_command = 'b'
                            speed = 30
                        elif 20 <= distance_scale <= 30:
                            distance_command = 's'
                            speed = 0
                        elif 30 < distance_scale <= 40:
                            distance_command = 'f'
                            speed = 20
                        elif 40 < distance_scale <= 50:
                            distance_command = 'f'
                            speed = 25
                        elif 50 < distance_scale <= 60:
                            distance_command = 'f'
                            speed = 30
                        elif 60 < distance_scale <= 70:
                            distance_command = 'f'
                            speed = 40
                        elif 70 < distance_scale <= 80:
                            distance_command = 'f'
                            speed = 50
                        elif 80 < distance_scale:
                            distance_command = 'f'
                            speed = 60

                        # 회전 제어
                        if abs(turn_direc) > TURN_MIN_VALUE: # 중심에서 벗어남
                            if turn_direc < 0: # 좌회전
                                turn_command = 'l'
                                speed = 30
                            else: # 우회전
                                turn_command = 'r'
                                speed = 30
                        else:
                            turn_command = 's'

                        # 명령 전송
                        if distance_command == 's' and turn_command == 's':   # 정지 + 회전x
                            send_command('s', speed)
                            print("정지")
                        elif distance_command == 'f' and turn_command == 's': # 전진 + 회전x
                            send_command('f', speed)
                            print("전진")
                        elif distance_command == 'f' and turn_command == 'l': # 전진 + 좌회전
                            send_command('z', speed)
                            print("전진 + 좌회전")
                        elif distance_command == 'f' and turn_command == 'r': # 전진 + 우회전
                            send_command('x', speed)
                            print("전진 + 우회전")
                        elif distance_command == 'b' and turn_command == 's': # 후진 + 회전x
                            send_command('b', speed)
                            print("후진")
                        elif distance_command == 'b' and turn_command == 'l': # 후진 + 좌회전
                            send_command('c', speed)
                            print("후진 + 좌회전")
                        elif distance_command == 'b' and turn_command == 'r': # 후진 + 우회전
                            send_command('v', speed)
                            print("후진 + 우회전")
                        elif distance_command == 's' and turn_command == 'l': # 정지 + 좌회전
                            send_command('l', speed)
                            print("정지 + 좌회전")
                        elif distance_command == 's' and turn_command == 'r': # 정지 + 우회전
                            send_command('r', speed)
                            print("정지 + 우회전")
                        else:
                            print("모르는 명령")
                    

                # UI 업데이트
                
                cv2.circle(frame, (frame_center_x, frame_center_y), 5, (0, 255, 0), cv2.FILLED)
                cv2.circle(frame, (box_center_x, box_center_y), 5, (255, 0, 0), cv2.FILLED)  # Bounding box center point
                cv2.putText(frame, f"Distance :  {distance_scale}", (20, 180), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 2)
                cv2.putText(frame, f"Turn Offset Value :  {turn_direc}", (20, 80), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 2)
                cv2.putText(frame, f"fps :  {fps}", (20, 20), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 2)
                cv2.line(frame, (box_center_x, box_center_y), (frame_center_x, frame_center_y), (255 ,0, 255), 2)
                cv2.putText(img=frame, text=f"Id: {track_id}", org=(xmin, ymin - 10), fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=2, color=(0, 255, 0), thickness=2)
                cv2.rectangle(img=frame, pt1=(xmin, ymin), pt2=(xmax, ymax), color=(0, 255, 0), thickness=2)

    else:
        print("휠체어 사용자가 탐지되지 않았습니다.")
        #send_command('s', pwm_distance, pwm_turn)
    #     current_time = time.time()
    #     if not is_sound_playing and (current_time - last_sound_time >= 5):
    #         mp3_file = '/home/soeunan/Kartriders/sounds/4.mp3'
    #         threading.Thread(target=play_sound, args=(mp3_file,), daemon=True).start()
    #         last_sound_time = current_time
    
    # # 60초마다 2.mp3 재생 (다른 음성이 재생 중이면 스킵)
    # if not is_sound_playing and (time.time() - sound_time_2min >= 60):
    #     mp3_file = '/home/soeunan/Kartriders/sounds/2.mp3'
    #     threading.Thread(target=play_sound, args=(mp3_file,), daemon=True).start()
    #     sound_time_2min = time.time()

    # 항상 카메라 화면을 표시
    cv2.imshow("Tarrrrmi", frame)

    # Press key q to stop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break



send_command('s')  # 멈춰
ser.close()

#종료 멘트
# mp3_file = '/home/soeunan/Kartriders/sounds/close.mp3'
# play_sound(mp3_file)

cap.release()
cv2.destroyAllWindows()