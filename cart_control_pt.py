from ultralytics import YOLO
import tflite_runtime.interpreter as tflite
import time
import cv2
import threading
#import serial
import pygame

print(">>> 카트 제어 스크립트 시작됨")

# pTime = 0
TURN_MIN_VALUE = 75

DISTANCE_MIN_VALUE = 10

is_sound_playing = False
last_sound_time = 0
sound_time_2min = time.time()

# Bluetooth 포트 설정
#arduino_port = '/dev/ttyACM0'  # 실제 포트에 맞게 수정해야 함
baudrate = 115200 # 이거 아두이노에 설정한 값이랑 같아야 한다.

# 시리얼 연결 설정
# ser = serial.Serial(arduino_port, baudrate) # 블루투스 장치와 연결

# # 명령 전송
"""
def send_command(char, num1=0): # send_command는 블루투스를 통해 명령을 아두이노로 전송하는 함수.\
    # num의 기본 값은 0으로 설정해서 정지할 때는 pwm 값을 아두이노로 안 보내도 된다.
    signal = f"{char},{num1}\n" # 문자와 숫자를 쉼표로 구분해서 전송. 
    ser.write(signal.encode()) # command 문자열을 바이트 형식으로 변환하고 ser을 통해 전송.
    print(f"명령 '{signal}' 전송됨")
"""
def play_sound(mp3_file):
    pygame.mixer.init()
    pygame.mixer.music.load(mp3_file)
    pygame.mixer.music.play()
    is_sound_playing = True
    while pygame.mixer.music.get_busy(): #play 중이면
        pygame.time.Clock().tick(10) #루프가 10초에 한 번씩 돌게 함
    is_sound_playing = False

#음성 재생동안 코드가 진행되지 않게 대기하는 함수        
def wait_with_time(duration):
    start_time = time.time()
    while time.time() - start_time < duration:
        time.sleep(0.01)

mp3_file = '/home/kart/yolo_test/sounds/intro.mp3'
play_sound(mp3_file)

# Initialize the webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("웹캠 초기화 실패. 종료합니다.")
    exit()

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 30)

# Initialize YOLOv8 object detector
model_path = "/home/kart/yolo_test/best.pt"
model = YOLO(model_path)

# Initialize OpenCV Tracker
tracker = cv2.TrackerCSRT_create()  # Use CSRT tracker
tracking = False  # Flag to check if we are tracking an object
bbox = None  # Bounding box for the object
frame_count = 0  # Frame counter for periodic YOLO updates

def calculate_iou(box1, box2):
    """두 박스의 IoU 계산."""
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    # 두 박스의 우측 하단 좌표 계산
    x1_end, y1_end = x1 + w1, y1 + h1
    x2_end, y2_end = x2 + w2, y2 + h2

    # 교차 영역의 좌상단과 우하단 좌표 계산
    x_intersect = max(x1, x2)
    y_intersect = max(y1, y2)
    x_intersect_end = min(x1_end, x2_end)
    y_intersect_end = min(y1_end, y2_end)

    # 교차 영역의 넓이 계산
    intersection_width = max(0, x_intersect_end - x_intersect)
    intersection_height = max(0, y_intersect_end - y_intersect)

    intersection_area = intersection_width * intersection_height

    # 두 박스의 넓이 계산
    area1 = w1 * h1
    area2 = w2 * h2

    # IOU 계산
    union_area = area1 + area2 - intersection_area
    iou = intersection_area / union_area if union_area > 0 else 0

    return iou

# Function to handle object detection and tracking in parallel (for performance)
def detect_and_track(frame):
    global bbox, tracking, tracker, frame_count

    frame_count += 1

    if not tracking or frame_count % 10 == 0:  # Use YOLO every 10 frames or when not tracking
        results = model.predict(source=frame, conf=0.7, iou=0.7, save=False, verbose=False)
        detections = results[0]

        if len(detections) > 0:
            boxes = detections.boxes.xyxy.cpu().numpy()
            class_ids = detections.boxes.cls.cpu().numpy()

            # Flag to track the first detected wheelchair
            detected = False

            for box, class_id in zip(boxes, class_ids):
                if int(class_id) == 1:  # Check for wheelchair user class
                    xmin, ymin, xmax, ymax = map(int, box)
                    detected_bbox = (xmin, ymin, xmax - xmin, ymax - ymin)

                    if tracking:
                        # Calculate IoU between YOLO and tracker bbox
                        iou = calculate_iou(detected_bbox, bbox)
                        if iou < 0.7:  # IoU threshold for reinitialization (improve threshold)
                            print("Low IoU. Gradually reinitializing tracker with YOLO detection.")
                            
                            del tracker
                            
                            tracker = cv2.TrackerCSRT_create()  # Create a new tracker
                            tracker.init(frame, detected_bbox)  # Reinitialize tracker
                            bbox = detected_bbox
                        else:
                            print("IoU sufficient. Keeping tracker bbox.")
                    else:
                        tracker.init(frame, detected_bbox)
                        bbox = detected_bbox
                        print("Tracking initialized.")

                    tracking = True
                    detected = True
                    break  # Only track the first detected wheelchair

            if not detected:
                print("No wheelchair user detected.")
                
        else:
            print("No detections found.")

    if tracking:
        success, updated_bbox = tracker.update(frame)
        if success:
            bbox = updated_bbox
            xmin, ymin, w, h = map(int, bbox)
            xmax = xmin + w
            ymax = ymin + h

            # Calculate the center of the frame
            box_center_x = xmin + w // 2
            box_center_y = ymin + h // 2
            frame_h, frame_w, _ = frame.shape
            frame_center_x = frame_w // 2
            frame_center_y = frame_h // 2
            turn_direc = box_center_x - frame_center_x
            distance_scale = frame_h - ymax

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
                if turn_direc < -150 : # 좌회전
                    turn_command = 'l1'
                    speed = 30
                elif -150 < turn_direc < 0 :
                    turn_command = 'l'
                    speed = 30
                elif 0 < turn_direc < 150 :
                    turn_command = 'r'
                    speed = 30
                else: # 우회전
                    turn_command = 'r1'
                    speed = 30
            else:
                turn_command = 's'

            # 명령 전송
            if distance_command == 's' and turn_command == 's':   # 정지 + 회전x
                #send_command('s', speed)
                print("정지")
            elif distance_command == 'f' and turn_command == 's': # 전진 + 회전x
                #send_command('f', speed)
                print("전진")
            elif distance_command == 'f' and turn_command == 'l': # 전진 + 좌회전
                #send_command('z', speed)
                print("전진 + 좌회전")
            elif distance_command == 'f' and turn_command == 'r': # 전진 + 우회전
                #send_command('x', speed)
                print("전진 + 우회전")
            elif distance_command == 'b' and turn_command == 's': # 후진 + 회전x
                #send_command('b', speed)
                print("후진")
            elif distance_command == 'b' and turn_command == 'l': # 후진 + 좌회전
                #send_command('c', speed)
                print("후진 + 좌회전")
            elif distance_command == 'b' and turn_command == 'r': # 후진 + 우회전
                #send_command('v', speed)
                print("후진 + 우회전")
            elif distance_command == 's' and turn_command == 'l': # 정지 + 좌회전
                #send_command('l', speed)
                print("정지 + 좌회전")
            elif distance_command == 's' and turn_command == 'r': # 정지 + 우회전
                #send_command('r', speed)
                print("정지 + 우회전")
            # elif turn_command == 'l1': # 전진 + 좌회전
            #     #send_command('f', speed)
            #     #time.sleep(2) #2초 기다린 후,
            #     #send_command('l', speed)
            #     print("2초 전진 후 좌회전")
                
            # elif turn_command == 'r1': # 전진 + 우회전
            #     #send_command('f', speed)
            #     #time.sleep(2) #2초 기다린 후,
            #     #send_command('r', speed)
            #     print("2초 전진 후 우회전")
            else:
                print("모르는 명령")
            
            # UI
            cv2.circle(frame, (frame_center_x, frame_center_y), 5, (0, 255, 0), cv2.FILLED)
            cv2.circle(frame, (box_center_x, box_center_y), 5, (255, 0, 0), cv2.FILLED)  # Bounding box center point
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (255, 0, 0), 2)  # Bounding box
            cv2.putText(frame, f"Distance :  {distance_scale}", (20, 180), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 2)
            cv2.putText(frame, f"Turn Offset Value :  {turn_direc}", (20, 80), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 2)
            cv2.putText(frame, f"fps :  {fps}", (20, 20), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 2)
            cv2.line(frame, (box_center_x, box_center_y), (frame_center_x, frame_center_y), (255 ,0, 255), 2)
        else:
            print("탐지 실패. 재탐색...")
            current_time = time.time()
            if not is_sound_playing and (current_time - last_sound_time >= 5):
                last_sound_time = current_time
                mp3_file = '/home/kart/yolo_test/sounds/detection.mp3'
                threading.Thread(target=play_sound, args=(mp3_file,), daemon=True).start()    
            #send_command('s')
            tracking = False

    return frame
    
pTime = 0
mp3_file = '/home/kart/yolo_test/sounds/ready.mp3'
play_sound(mp3_file)

# Main roop
try:
    while True:
        # Capture frame from camera
        ret, frame = cap.read()
        if not ret:
            print("카메라 연결 실패.. 종료.")
            break
        
        
        # Calculate FPS
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
    

        # Perform object detection and tracking
        frame = detect_and_track(frame)

        # Display the frame
        cv2.imshow("Detected Objects", frame)

        # Press key 'q' to stop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Exit key pressed.")
            break
        #60초마다 2.mp3 재생 (다른 음성이 재생 중이면 스킵)
        if not is_sound_playing and (time.time() - sound_time_2min >= 60):
            mp3_file = '/home/kart/yolo_test/sounds/warning.mp3'
            threading.Thread(target=play_sound, args=(mp3_file,), daemon=True).start()
            sound_time_2min = time.time()


except KeyboardInterrupt: # Ctrl + C 눌러서 정지했을 때
    print("\n강제 종료.")
finally:
    # 프로그램이 어떤 이유로 종료되든, 반드시 실행되는 코드
    #send_command('s')  # 멈춰
    #종료 멘트
    mp3_file = '/home/kart/yolo_test/sounds/warning.mp3'
    play_sound(mp3_file)
    #ser.close()  # 블루투스 연결 종료
    cap.release()  # 카메라 종료
    cv2.destroyAllWindows()  # opencv 창 종료
    print("리소스 해제. 프로그램 종료.")
