import cv2
import numpy as np
import os
import time
from PIL import ImageFont, ImageDraw, Image

# CẤU HÌNH
VIDEO_PATH = "Recorded_Videos"
FRAME_COUNT_LIMIT = 30
FPS = 20

os.makedirs(VIDEO_PATH, exist_ok=True)

# HÀM HIỂN THỊ TEXT TIẾNG VIỆT
def put_vietnamese_text(img, text, position, font_size=30, color=(0,255,0)):
    
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)

    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except:
        font = ImageFont.load_default()

    draw.text(position, text, font=font, fill=color)

    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

# LẤY INDEX VIDEO LỚN NHẤT
def get_start_index(folder, action):

    indices = []

    for file in os.listdir(folder):

        if file.endswith(".avi") and file.startswith(action+"_"):

            try:
                index = int(file.replace(".avi","").split("_")[-1])
                indices.append(index)

            except:
                pass

    if len(indices) == 0:
        return 0

    return max(indices) + 1

# NHẬP THÔNG TIN
action = input("Nhập tên hành động: ")

no_sequences = int(input(f"Số lượng video cần quay cho '{action}': "))

action_video_dir = os.path.join(VIDEO_PATH, action)

os.makedirs(action_video_dir, exist_ok=True)

start_index = get_start_index(action_video_dir, action)

print(f"Bắt đầu quay từ index: {start_index}")

# CAMERA
cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FRAME_WIDTH,640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT,480)

frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

fourcc = cv2.VideoWriter_fourcc(*'XVID')

started = False
should_quit = False

print("Điều khiển:")
print("S - bắt đầu quay")
print("P - dừng quay")
print("Q - thoát")

# VÒNG LẶP CHÍNH
while cap.isOpened():

    ret, frame = cap.read()

    if not ret or should_quit:
        break

    frame = cv2.flip(frame,1)

    # CHỜ BẮT ĐẦU
    if not started:

        display = frame.copy()

        display = put_vietnamese_text(display,f"NHÃN: {action}",(20,20),30,(255,255,255))

        display = put_vietnamese_text(display,"NHẤN S ĐỂ BẮT ĐẦU",(150,200),40,(0,255,255))

        display = put_vietnamese_text(display,"NHẤN Q ĐỂ THOÁT",(20,440),20,(0,0,255))

        cv2.imshow("Record Dataset",display)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('s'):
            started = True

        if key == ord('q'):
            should_quit = True

        continue

    # QUAY VIDEO
    for sequence in range(start_index,start_index + no_sequences):

        if not started or should_quit:
            break

        for count in range(3,0,-1):

            start_time = time.time()

            while time.time() - start_time < 1:

                ret, frame = cap.read()

                frame = cv2.flip(frame,1)

                img = frame.copy()

                img = put_vietnamese_text(img,f"CHUẨN BỊ: {count}",(220,200),50,(0,165,255))

                cv2.imshow("Record Dataset",img)

                key = cv2.waitKey(1) & 0xFF

                if key == ord('p'):
                    started = False
                    break

                if key == ord('q'):
                    should_quit = True
                    break

            if not started or should_quit:
                break

        if not started or should_quit:
            break

        # TẠO VIDEO
        filename = os.path.join(action_video_dir,f"{action}_{sequence}.avi")

        out = cv2.VideoWriter(filename,fourcc,FPS,(frame_width,frame_height))

        # GHI FRAME
        for f in range(FRAME_COUNT_LIMIT):

            ret, frame = cap.read()

            if not ret:
                break

            frame = cv2.flip(frame,1)

            out.write(frame)

            display = frame.copy()

            display = put_vietnamese_text(display,f"ĐANG QUAY: {action}",(20,20),30,(0,255,0))

            display = put_vietnamese_text(display,f"VIDEO: {sequence}",(20,60),25,(0,255,255))

            display = put_vietnamese_text(display,f"FRAME: {f+1}/{FRAME_COUNT_LIMIT}",(20,100),25,(255,255,255))

            cv2.imshow("Record Dataset",display)

            key = cv2.waitKey(1) & 0xFF

            if key == ord('p'):
                started = False
                break

            if key == ord('q'):
                should_quit = True
                break

        out.release()

        # NGHỈ 1S
        pause = time.time()

        while time.time() - pause < 1:

            ret, frame = cap.read()

            frame = cv2.flip(frame,1)

            img = frame.copy()

            img = put_vietnamese_text(img,"NGHỈ 1S",(250,200),40,(0,0,255))

            cv2.imshow("Record Dataset",img)

            key = cv2.waitKey(1) & 0xFF

            if key == ord('p'):
                started = False
                break

            if key == ord('q'):
                should_quit = True
                break

    # reset
    if not should_quit:

        print(f"Đã quay xong {no_sequences} video cho '{action}'")

        start_index += no_sequences

        started = False

cap.release()
cv2.destroyAllWindows()