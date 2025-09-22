import cv2
import pyrealsense2 as rs
import numpy as np
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator

# 1. RealSense 파이프라인 설정
pipeline = rs.pipeline()
config = rs.config()

# 스트림 설정 (컬러)
# 사용 가능한 해상도와 프레임률은 RealSense Viewer에서 확인 가능
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# 파이프라인 시작
pipeline.start(config)

# YOLO 모델 로드
model = YOLO("best.pt")

try:
    while True:
        # 2. RealSense에서 프레임 가져오기
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        
        # 프레임이 없으면 다음 루프로
        if not color_frame:
            continue

        # 프레임을 OpenCV에서 사용할 수 있는 numpy 배열로 변환
        frame = np.asanyarray(color_frame.get_data())

        # 3. 기존 YOLO 추론 및 시각화 코드 (이 부분은 동일)
        results = model(frame, verbose=False)
        boxes = results[0].boxes

        if boxes is not None:
            conf = boxes.conf
            keep_idx = conf > 0.5  # confidence 0.5 이상만
            kept_boxes = boxes[keep_idx]

            annotator = Annotator(frame)
            for box in kept_boxes:
                b = box.xyxy[0].cpu().numpy().astype(int)
                c = int(box.cls)
                conf_score = float(box.conf)
                label = f"{model.names[c]} {conf_score:.2f}"
                annotator.box_label(b, label)

            frame = annotator.result()

        cv2.imshow("Realsense Test", frame)
        key = cv2.waitKey(1)
        if key == ord('q'):
            break

finally:
    # 4. 파이프라인 중지
    pipeline.stop()
    cv2.destroyAllWindows()