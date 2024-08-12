import cv2 as cv
import mediapipe as mp
import numpy as np
import time
from PIL import Image
#import viscosity_sensor  # Replace with the actual module name
#import brake_fluid_sensor  # Replace with the actual module name

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

def load_glove_image(image_path):
    glove_img = Image.open(image_path)
    if glove_img.mode != 'RGBA':
        glove_img = glove_img.convert('RGBA')
    glove_img = np.array(glove_img)
    return glove_img

glove_img = load_glove_image('C:\\hand-gesture-recognition-mediapipe-main\\glove.png')
glove_height, glove_width = glove_img.shape[:2]

def overlay_image_alpha(img, img_overlay, pos, alpha_mask):
    x, y = pos
    y1, y2 = max(0, y), min(img.shape[0], y + img_overlay.shape[0])
    x1, x2 = max(0, x), min(img.shape[1], x + img_overlay.shape[1])
    y1o, y2o = max(0, -y), min(img_overlay.shape[0], img.shape[0] - y)
    x1o, x2o = max(0, -x), min(img_overlay.shape[1], img.shape[1] - x)
    if y1 >= y2 or x1 >= x2 or y1o >= y2o or x1o >= x2o:
        return
    img_slice = img[y1:y2, x1:x2]
    overlay_slice = img_overlay[y1o:y2o, x1o:x2o]
    alpha = alpha_mask[y1o:y2o, x1o:x2o, None] / 255.0
    img_slice[:] = alpha * overlay_slice[:, :, :3] + (1.0 - alpha) * img_slice

def calculate_angle(p1, p2, p3):
    """Calculate the angle between three points."""
    a = np.array(p1)
    b = np.array(p2)
    c = np.array(p3)
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180.0:
        angle = 360 - angle
    return angle

def detect_hand_gesture(hand_landmarks):
    """Detect if a hand is making a fist or other gesture based on landmark angles."""
    gestures = {}
    for finger, tips in [("thumb", [mp_hands.HandLandmark.THUMB_TIP, mp_hands.HandLandmark.THUMB_IP]),
                         ("index", [mp_hands.HandLandmark.INDEX_FINGER_TIP, mp_hands.HandLandmark.INDEX_FINGER_DIP]),
                         ("middle", [mp_hands.HandLandmark.MIDDLE_FINGER_TIP, mp_hands.HandLandmark.MIDDLE_FINGER_DIP]),
                         ("ring", [mp_hands.HandLandmark.RING_FINGER_TIP, mp_hands.HandLandmark.RING_FINGER_DIP]),
                         ("pinky", [mp_hands.HandLandmark.PINKY_TIP, mp_hands.HandLandmark.PINKY_DIP])]:
        
        angle = calculate_angle(
            (hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].x, hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].y),
            (hand_landmarks.landmark[tips[1]].x, hand_landmarks.landmark[tips[1]].y),
            (hand_landmarks.landmark[tips[0]].x, hand_landmarks.landmark[tips[0]].y)
        )
        gestures[finger] = angle < 90  # Finger considered bent if angle < 90 degrees
    return gestures

#def draw_viscosity_info(image, viscosity_value):
    #cv.putText(image, f"Viscosity: {viscosity_value}", (10, 120), cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv.LINE_AA)
    #return image

#def draw_brake_fluid_info(image, brake_fluid_value, pos):
   # """Overlay the brake fluid level at the given position."""
    #cv.putText(image, f"Brake Fluid: {brake_fluid_value}%", pos, cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv.LINE_AA)
   # return image

def main():
    cap = cv.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Webcam could not be opened.")
        return

    hands = mp_hands.Hands()
    fps_counter = 0
    start_time = time.time()
    image_counter = 0
    save_interval = 10
    last_save_time = time.time()

    # Initialize the viscosity sensor
    #sensor = viscosity_sensor.ViscositySensor()  # Replace with the actual initialization method
    
    # Initialize the brake fluid sensor
    #brake_sensor = brake_fluid_sensor.BrakeFluidSensor()  # Replace with the actual initialization method

    while cap.isOpened():
        success, img = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        results = hands.process(img_rgb)

        h, w, c = img.shape
        debug_image = img.copy()

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Detect hand gesture
                gestures = detect_hand_gesture(hand_landmarks)
                print("Detected Gestures:", gestures)

                wrist_x = int(hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].x * img.shape[1])
                wrist_y = int(hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].y * img.shape[0])
                middle_finger_tip_x = int(hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].x * img.shape[1])
                middle_finger_tip_y = int(hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y * img.shape[0])

                cv.circle(debug_image, (wrist_x, wrist_y), 10, (255, 0, 0), -1)
                cv.circle(debug_image, (middle_finger_tip_x, middle_finger_tip_y), 10, (0, 0, 255), -1)

                dx = middle_finger_tip_x - wrist_x
                dy = middle_finger_tip_y - wrist_y
                angle = np.degrees(np.arctan2(dy, dx)) + 90

                glove_scale = np.sqrt(dx * dx + dy * dy) / glove_height
                M = cv.getRotationMatrix2D((glove_width // 2, glove_height // 2), -angle, glove_scale)
                rotated_glove = cv.warpAffine(glove_img, M, (glove_width, glove_height), flags=cv.INTER_LINEAR, borderMode=cv.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))

                if rotated_glove.shape[2] == 4:
                    glove_alpha = rotated_glove[:, :, 3]
                    glove_overlay = rotated_glove[:, :, :3]
                else:
                    glove_alpha = np.ones((glove_height, glove_width), dtype=np.uint8) * 255
                    glove_overlay = rotated_glove

                glove_x = middle_finger_tip_x - int(glove_width * glove_scale / 2)
                glove_y = middle_finger_tip_y - int(glove_height * glove_scale / 2)

                overlay_image_alpha(img, glove_overlay, (glove_x, glove_y), glove_alpha)

                # Get index finger tip position
                index_finger_tip_x = int(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * img.shape[1])
                index_finger_tip_y = int(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * img.shape[0])

                # Read viscosity value from the sensor
                #viscosity_value = sensor.read_viscosity()
                #print(f"Viscosity at Index Finger Tip: {viscosity_value}")

                # Read brake fluid value from the sensor
                #brake_fluid_value = brake_sensor.read_brake_fluid()
                #print(f"Brake Fluid Level at Index Finger Tip: {brake_fluid_value}")

                # Draw viscosity info on the image
                #img = draw_viscosity_info(img, viscosity_value)

                # Draw brake fluid level on the image
                #img = draw_brake_fluid_info(img, brake_fluid_value, (index_finger_tip_x + 10, index_finger_tip_y - 10))

                landmark_list = calc_landmark_list(img, hand_landmarks)
                brect = calc_bounding_rect(img, hand_landmarks)

                img = draw_bounding_rect(img, brect)
                img = draw_landmarks(img, landmark_list)

        cv.imshow('Image', debug_image)
        cv.imshow("Hand with Glove", img)

        fps_counter += 1
        if time.time() - start_time >= 1:
            fps = fps_counter
            fps_counter = 0
            start_time = time.time()
        else:
            fps = 0

        img = draw_info(img, fps, "default", 0)

        current_time = time.time()
        if current_time - last_save_time >= save_interval:
            image_filename = f"captured_image_{image_counter}.jpg"
            cv.imwrite(image_filename, img)
            print(f"Saved {image_filename}")
            image_counter += 1
            last_save_time = current_time

        if cv.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()

def calc_bounding_rect(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]
    landmark_array = np.empty((0, 2), int)
    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        landmark_point = [np.array((landmark_x, landmark_y))]
        landmark_array = np.append(landmark_array, landmark_point, axis=0)
    x, y, w, h = cv.boundingRect(landmark_array)
    return [x, y, x + w, y + h]

def calc_landmark_list(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]
    landmark_list = []
    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        landmark_list.append([landmark_x, landmark_y])
    return landmark_list

def draw_bounding_rect(image, brect):
    cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[3]), (0, 255, 0), 2)
    return image

def draw_landmarks(image, landmark_list):
    for landmark in landmark_list:
        cv.circle(image, tuple(landmark), 5, (0, 0, 255), -1)
    return image

def draw_info(image, fps, mode, number):
    cv.putText(image, "FPS:" + str(fps), (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv.LINE_AA)
    cv.putText(image, "MODE:" + str(mode), (10, 60), cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv.LINE_AA)
    cv.putText(image, "NUM:" + str(number), (10, 90), cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv.LINE_AA)
    return image

if __name__ == '__main__':
    main()
