import cv2
import numpy as np
import os
import sys
import mediapipe as mp
from tensorflow.keras.models import load_model
from PIL import ImageFont, ImageDraw, Image
import platform
import unicodedata

MODELS_CONFIG = {
    "action": {
        "model": "models/model_actions_alpha.keras",
        "labels": "models/actions_alpha_list.npy",
        "name": "HÀNH ĐỘNG"
    },
    "number": {
        "model": "models/model_numbers.keras",
        "labels": "models/number_list.npy",
        "name": "CHỮ SỐ"
    }
}

SEQUENCE_LENGTH = 30
THRESHOLD = 0.8

model = None
actions = None
current_mode = "action"

sequence = []

# LOAD MODEL
def load_system_model(mode_type):

    global model, actions, current_mode

    config = MODELS_CONFIG[mode_type]

    print(f"\nLoading {config['name']} model...")

    try:
        model = load_model(config["model"])
        actions = np.load(config["labels"])

        current_mode = mode_type

        print("Loaded successfully")
        print("Classes:", actions)

        return True

    except Exception as e:

        print("Error loading model:", e)
        return False


if not load_system_model("action"):
    sys.exit()

FONT_PATH = "arial.ttf"

if platform.system() == "Windows":

    fonts = [
        "C:/Windows/Fonts/segoeui.ttf",
        "C:/Windows/Fonts/tahoma.ttf",
        "C:/Windows/Fonts/arial.ttf"
    ]

    for f in fonts:
        if os.path.exists(f):
            FONT_PATH = f
            break


def put_text_vietnamese(img, text, pos=(20,20), size=30, color=(255,255,255)):

    text = unicodedata.normalize("NFC", text)

    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    draw = ImageDraw.Draw(img_pil)

    try:
        font = ImageFont.truetype(FONT_PATH, size)
    except:
        font = ImageFont.load_default()

    draw.text(pos, text, font=font, fill=color)

    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

# MEDIAPIPE
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

# KEYPOINT EXTRACTION
def extract_keypoints(results):

    if results.left_hand_landmarks:

        l_res = np.array(
            [[res.x, res.y, res.z]
             for res in results.left_hand_landmarks.landmark]
        )

        l_res = l_res - l_res[0]

        lh = l_res.flatten()

    else:
        lh = np.zeros(21 * 3)

    if results.right_hand_landmarks:

        r_res = np.array(
            [[res.x, res.y, res.z]
             for res in results.right_hand_landmarks.landmark]
        )

        r_res = r_res - r_res[0]

        rh = r_res.flatten()

    else:
        rh = np.zeros(21 * 3)

    return np.concatenate([lh, rh])

# MEDIAPIPE DETECTION
def mediapipe_detection(image, model):

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    image.flags.writeable = False

    results = model.process(image)

    image.flags.writeable = True

    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    return image, results

# CAMERA
cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FRAME_WIDTH,640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT,480)

last_msg = "Sẵn sàng..."
last_color = (200,200,200)

frame_counter = 0

# MAIN LOOP
with mp_holistic.Holistic(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        model_complexity=1
) as holistic:

    while cap.isOpened():

        ret, frame = cap.read()

        if not ret:
            break

        frame = cv2.flip(frame,1)

        image, results = mediapipe_detection(frame, holistic)

        # Draw landmarks

        # mp_drawing.draw_landmarks(
        #     image,
        #     results.left_hand_landmarks,
        #     mp_holistic.HAND_CONNECTIONS
        # )

        # mp_drawing.draw_landmarks(
        #     image,
        #     results.right_hand_landmarks,
        #     mp_holistic.HAND_CONNECTIONS
        # )

        # PREDICTION (PIPELINE 1 STYLE)
        if results.left_hand_landmarks or results.right_hand_landmarks:

            keypoints = extract_keypoints(results)

            sequence.append(keypoints)

            sequence = sequence[-SEQUENCE_LENGTH:]

            frame_counter += 1

            if len(sequence) == SEQUENCE_LENGTH and frame_counter % 5 == 0:

                res = model.predict(
                    np.expand_dims(sequence, axis=0),
                    verbose=0
                )[0]

                idx = np.argmax(res)

                confidence = res[idx]

                if confidence > THRESHOLD:

                    action_name = actions[idx]

                    last_msg = f"[{MODELS_CONFIG[current_mode]['name']}] {action_name} ({confidence:.0%})"

                    last_color = (0,255,0)

        # UI
        cv2.rectangle(image,(0,0),(640,80),(0,0,0),-1)

        image = put_text_vietnamese(
            image,
            last_msg,
            (20,10),
            28,
            last_color
        )

        help_text = "A: Hành động | N: Chữ số | Q: Thoát"

        image = put_text_vietnamese(
            image,
            help_text,
            (20,45),
            18,
            (255,255,0)
        )

        cv2.imshow("Nhan dien ngon ngu ky hieu", image)

        # KEYBOARD
        key = cv2.waitKey(10) & 0xFF

        if key == ord("q"):
            break

        elif key == ord("a"):

            if current_mode != "action":

                if load_system_model("action"):

                    sequence = []
                    frame_counter = 0
                    last_msg = "Đã chuyển sang Hành động"

        elif key == ord("n"):

            if current_mode != "number":

                if load_system_model("number"):

                    sequence = []
                    frame_counter = 0
                    last_msg = "Đã chuyển sang Chữ số"


cap.release()
cv2.destroyAllWindows()