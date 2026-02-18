# python3 sort_coral_tpu.py {UUID} --debug

import cv2
import time
import threading
import serial
from ultralytics import YOLO
from short import Sort
import numpy as np
import os
from datetime import datetime
from collections import Counter
import requests
import sys
from signal import signal, SIGTERM
import atexit
import pygame

DEBUG_MODE = "--debug" in sys.argv
STATUS_FILE = "/tmp/cart_alive"

def send_command_out(char, num1=0):
        signal = f"{char},{num1}\n"
        ser.write(signal.encode())
        print(f"명령 '{signal}' 전송됨")

def create_status_file():
    with open(STATUS_FILE, "w") as f:
        f.write("ready")

def remove_status_file():
    if os.path.exists(STATUS_FILE):
        os.remove(STATUS_FILE)

atexit.register(remove_status_file)

serial_port = '/dev/ttyACM0'
baudrate = 9600
ser = serial.Serial(serial_port, baudrate)

def clean_exit():
    wheelchair_thread.stop()
    product_thread.stop()
    ser.close()
    try:
        pygame.mixer.init()
        pygame.mixer.music.load('/home/soeunan/Kartriders/sounds/close.mp3')
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(10)
    except:
        pass
    exit(0)

global is_sound_playing
is_sound_playing = False
# global last_sound_time
# last_sound_time = 0
# global sound_time_2min
# sound_time_2min = time.time()

# 음성 출력 함수
def play_sound(mp3_file):
    pygame.mixer.init()
    pygame.mixer.music.load(mp3_file)
    pygame.mixer.music.play()
    is_sound_playing = True
    while pygame.mixer.music.get_busy(): #play 중이면
        pygame.time.Clock().tick(10) #루프가 10초에 한 번씩 돌게 함
    is_sound_playing = False


def handle_sigterm(signum, frame):
    clean_exit()
    remove_status_file()

signal(SIGTERM, handle_sigterm)

class WheelchairDetectionThread(threading.Thread):
    def __init__(self, cam_index=0):
        super().__init__()
        self.cam_index = cam_index
        self.running = True
        self.is_sound_playing = False
        self.last_sound_time = time.time()
        self.sound_time_2min = time.time()

    def run(self):
        model = YOLO("/home/kart/yolo_test/best_full_integer_quant_edgetpu.tflite", task='detect')
        tracker = Sort(max_age=50, min_hits=3, iou_threshold=0.1)
        TURN_MIN_VALUE = 75

        cap = cv2.VideoCapture(self.cam_index, cv2.CAP_V4L2)
        if not cap.isOpened():
            print("휠체어 카메라 열기 실패")
            return
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 400)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 300)

        mp3_file = '/home/kart/yolo_test/sounds/ready.mp3' #ready
        play_sound(mp3_file)

        frame_count = 0

        while self.running:
            ret, frame = cap.read()
            if not ret:
                continue
            
            frame_h, frame_w, _ = frame.shape
            frame_center_x = frame_w // 2
            results = model.predict(source=frame, conf=0.4, iou=0.3, save=False, verbose=False)
            results = results[0]
            class_ids = results.boxes.cls.cpu().numpy()
            boxes = results.boxes.xyxy.cpu().numpy()
            frame_center_x = frame_w // 2
            frame_center_y = frame_h // 2
            

            if 0 in class_ids:
                for box, class_id in zip(boxes, class_ids):
                    if class_id == 0:
                       
                        tracks = tracker.update(boxes)
                        if len(tracks) > 0:
                            xmin, ymin, xmax, ymax, track_id = tracks[-1].astype(int)
                            box_center_x = xmin + (xmax - xmin) // 2
                            distance_scale = frame_h - ymax
                            turn_direc = frame_center_x - box_center_x

                            if frame_count % 10 == 0:
                                # 거리 제어
                                if distance_scale <= 19:
                                    distance_command = 'b'
                                    speed = 40
                                elif 20 <= distance_scale <= 30:
                                    distance_command = 's'
                                    speed = 0
                                elif 30 < distance_scale <= 40:
                                    distance_command = 'f'
                                    speed = 50
                                elif 40 < distance_scale <= 50:
                                    distance_command = 'f'
                                    speed = 55
                                elif 50 < distance_scale <= 60:
                                    distance_command = 'f'
                                    speed = 60
                                elif 60 < distance_scale <= 70:
                                    distance_command = 'f'
                                    speed = 70
                                elif 70 < distance_scale <= 80:
                                    distance_command = 'f'
                                    speed = 80
                                elif 80 < distance_scale:
                                    distance_command = 'f'
                                    speed = 90

                                # 회전 제어
                                if abs(turn_direc) > TURN_MIN_VALUE: # 중심에서 벗어남
                                    if turn_direc > 0: # 좌회전
                                        turn_command = 'l'
                                        # speed = 70
                                    else: # 우회전
                                        turn_command = 'r'
                                        # speed = 70
                                else:
                                    turn_command = 's'

                                self.send_combined_command(distance_command, turn_command, speed)
                        if DEBUG_MODE:
                            h = int(box[3]) - int(box[1])
                            box_center_y = ymin + h // 2
                            cv2.circle(frame, (frame_center_x, frame_center_y), 5, (0, 255, 0), cv2.FILLED)
                            cv2.circle(frame, (box_center_x, box_center_y), 5, (255, 0, 0), cv2.FILLED)  # Bounding box center point
                            cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 2)
                            cv2.putText(frame, "Wheelchair", (int(box[0]), int(box[1]) - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                            cv2.putText(frame, f"Distance :  {distance_scale}", (20, 180), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 2)
                            cv2.putText(frame, f"Turn Offset Value :  {turn_direc}", (20, 80), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 2)
           
            else:
                print("휠체어 사용자가 탐지되지 않았습니다.")
                self.send_command('s')

                current_time = time.time()
                if not self.is_sound_playing and (current_time - self.last_sound_time >= 5):
                    mp3_file = '/home/kart/yolo_test/sounds/detection.mp3'
                    threading.Thread(target=play_sound, args=(mp3_file,), daemon=True).start()
                    self.last_sound_time = current_time
            
            if not self.is_sound_playing and (time.time() - self.sound_time_2min >= 60):
                mp3_file = '/home/kart/yolo_test/sounds/warning.mp3'
                threading.Thread(target=play_sound, args=(mp3_file,), daemon=True).start()
                self.sound_time_2min = time.time()

            frame_count += 1

            if DEBUG_MODE:
                cv2.imshow("Wheelchair", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    self.stop()

        cap.release()

    def send_command(self, char, num1=0):
        signal = f"{char},{num1}\n"
        ser.write(signal.encode())
        print(f"명령 '{signal}' 전송됨")

    def send_combined_command(self, d_cmd, t_cmd, speed):
        mapping = {
            ('s','s'):'s', ('f','s'):'f', ('f','l'):'z', ('f','r'):'x',
            ('b','s'):'b', ('b','l'):'c', ('b','r'):'v', ('s','l'):'l', ('s','r'):'r'
        }
        cmd = mapping.get((d_cmd, t_cmd), 's')
        self.send_command(cmd, speed)

    def stop(self):
        self.running = False

class ProductDetectionThread(threading.Thread):
    def __init__(self, cam_index=2, cart_uuid=None):
        super().__init__()
        self.cam_index = cam_index
        self.cart_uuid = cart_uuid
        self.running = True

    def run(self):
        model = YOLO("/home/kart/yolo_test/best_kgh.pt", task="detect")
        previous_counter = Counter()

        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        save_dir = os.path.join(BASE_DIR, "detections", "annotated")
        os.makedirs(save_dir, exist_ok=True)

        cap = cv2.VideoCapture(self.cam_index, cv2.CAP_V4L2)
        if not cap.isOpened():
            print("상품 카메라 열기 실패")
            return

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)

        while self.running:
            ret, frame = cap.read()
            if not ret:
                time.sleep(1)
                continue

            results = model(frame, conf=0.5)
            class_ids = results[0].boxes.cls.tolist()
            names = results[0].names
            products_detected = [names[int(cls_id)] for cls_id in class_ids]
            current_counter = Counter(products_detected)

            new_products = []
            for item, count in current_counter.items():
                if count > previous_counter.get(item, 0):
                    diff = count - previous_counter.get(item, 0)
                    new_products.extend([item] * diff)

            if new_products:
                print("새 상품 발견:", new_products)
                self.send_to_server(new_products)
                previous_counter.update(new_products)

            annotated = results[0].plot()
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            image_path = os.path.join(save_dir, f"{timestamp}_annotated.jpg")
            cv2.imwrite(image_path, annotated)

            time.sleep(30)

        cap.release()

    def send_to_server(self, products):
        url = f"http://localhost:3000/cart/yolo_detect"
        payload = {"detected": products, "cart_uuid": self.cart_uuid}
        try:
            response = requests.post(url, json=payload)
            print("서버 응답:", response.status_code, response.text)
        except Exception as e:
            print("서버 전송 실패:", e)

    def stop(self):
        self.running = False

if __name__ == "__main__":
    cart_uuid = sys.argv[1]
    print(f"시작된 카트 UUID: {cart_uuid}")
    #sound_time_2min = time.time()

    mp3_file = '/home/kart/yolo_test/sounds/intro.mp3' #intro
    play_sound(mp3_file)

    wheelchair_thread = WheelchairDetectionThread(cam_index=0)
    product_thread = ProductDetectionThread(cam_index=2, cart_uuid=cart_uuid)
    

    wheelchair_thread.start()
    product_thread.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        clean_exit()
        remove_status_file()
        send_command_out('s')
    finally:
        mp3_file = '/home/kart/yolo_test/sounds/close.mp3' #close
        play_sound(mp3_file)
        send_command_out('s')