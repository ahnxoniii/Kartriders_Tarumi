import cv2
import time
import os
from datetime import datetime
from ultralytics import YOLO
from collections import Counter
import requests

# 서버 전송 함수
def send_to_server(products_detected):
    url = "http://localhost:8000/detect"
    payload = {"detected": products_detected}
    try:
        response = requests.post(url, json=payload)
        if response.status_code == 200:
            print("서버 전송 성공:", response.json())
        else:
            print("서버 응답 오류:", response.status_code)
    except Exception as e:
        print("서버 전송 실패:", e)

# 상품 인식 메인 함수
def run_product_detection(cam_index=2):
    # YOLO 모델 로드
    yolo_snacks = YOLO("/home/kart/yolo_test/best_kgh.pt")
    previous_counter = Counter()

    # 저장 경로 준비
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    save_dir = os.path.join(BASE_DIR, "detections", "annotated")
    os.makedirs(save_dir, exist_ok=True)

    # 카메라 한 번만 열고 계속 유지
    cap = cv2.VideoCapture(cam_index)
    if not cap.isOpened():
        print(f"카메라 {cam_index} 열기 실패")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("카메라 캡처 실패 (이미지 없음)")
                time.sleep(1)
                continue

            results = yolo_snacks(frame, conf=0.5)
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
                print("새로 추가된 상품:", new_products)
                send_to_server(new_products)
                previous_counter.update(new_products)
            else:
                print("새로 추가된 상품 없음")

            # 결과 이미지 저장
            annotated = results[0].plot()
            timestmp = datetime.now().strftime("%Y%m%d_%H%M%S")
            image_path = os.path.join(save_dir, f"{timestamp}_annotated.jpg")
            success = cv2.imwrite(image_path, annotated)
            if not success:
                print(f"이미지 저장 실패: {image_path}")

            time.sleep(5)  # 5초마다 반복 (조절 가능)

    except KeyboardInterrupt:
        print("사용자 종료 신호 수신")
    finally:
        cap.release()
        print("카메라 연결 해제. 프로그램 종료.")

if __name__ == "__main__":
    run_product_detection(cam_index=2)
