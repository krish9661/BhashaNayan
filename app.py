from threading import Timer
import webbrowser
from flask import Flask, flash, redirect, render_template, Response, request, session
import cv2 as cv
import numpy as np
from collections import deque, Counter
import copy
import csv
import itertools
import mediapipe as mp
from utils import CvFpsCalc
from model import KeyPointClassifier, PointHistoryClassifier
import os
from dotenv import load_dotenv
from flask_mysqldb import MySQL
from werkzeug.security import generate_password_hash, check_password_hash
import MySQLdb.cursors  # Make sure this is imported

load_dotenv()
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# Flask app setup
app = Flask(__name__, template_folder='templates', static_folder='static')
app.secret_key = os.getenv("SECRET_KEY")

# MySQL Config

app.config['MYSQL_HOST'] = os.getenv('MYSQL_HOST')
app.config['MYSQL_USER'] = os.getenv('MYSQL_USER')
app.config['MYSQL_PASSWORD'] = os.getenv('MYSQL_PASSWORD')
app.config['MYSQL_DB'] = 'bhashanayan'
app.config['MYSQL_CURSORCLASS'] = os.getenv('MYSQL_CURSORCLASS')

mysql = MySQL(app)

# Global variables
args = {
    "device": 0,
    "width": 960,
    "height": 540,
    "min_detection_confidence": 0.7,
    "min_tracking_confidence": 0.5,
}

# Initialize models and variables
def initialize_models():
    global hands, keypoint_classifier, point_history_classifier, history_length, point_history, finger_gesture_history

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=args["min_detection_confidence"],
        min_tracking_confidence=args["min_tracking_confidence"],
    )

    keypoint_classifier = KeyPointClassifier()
    point_history_classifier = PointHistoryClassifier()

    history_length = 16
    point_history = deque(maxlen=history_length)
    finger_gesture_history = deque(maxlen=history_length)

initialize_models()

# Load labels
with open("model/keypoint_classifier/keypoint_classifier_label.csv", encoding="utf-8-sig") as f:
    keypoint_classifier_labels = [row[0] for row in csv.reader(f)]

with open("model/point_history_classifier/point_history_classifier_label.csv", encoding="utf-8-sig") as f:
    point_history_classifier_labels = [row[0] for row in csv.reader(f)]


# Helper functions (as defined in your code, including pre-processing, drawing functions, etc.)

def calc_bounding_rect(image, landmarks):
    # Your bounding rect calculation function here
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
    # Your landmark calculation function here
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_point = []

    # Keypoint
    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        # landmark_z = landmark.z

        landmark_point.append([landmark_x, landmark_y])

    return landmark_point

def pre_process_landmark(landmark_list):
    # Your pre-processing function here
    temp_landmark_list = copy.deepcopy(landmark_list)

    # Convert to relative coordinates
    base_x, base_y = 0, 0
    for index, landmark_point in enumerate(temp_landmark_list):
        if index == 0:
            base_x, base_y = landmark_point[0], landmark_point[1]

        temp_landmark_list[index][0] = temp_landmark_list[index][0] - base_x
        temp_landmark_list[index][1] = temp_landmark_list[index][1] - base_y

    # Convert to a one-dimensional list
    temp_landmark_list = list(
        itertools.chain.from_iterable(temp_landmark_list))

    # Normalization
    max_value = max(list(map(abs, temp_landmark_list)))

    def normalize_(n):
        return n / max_value

    temp_landmark_list = list(map(normalize_, temp_landmark_list))

    return temp_landmark_list

def pre_process_point_history(image, point_history):
    # Your point history pre-processing function here
    image_width, image_height = image.shape[1], image.shape[0]

    temp_point_history = copy.deepcopy(point_history)

    # Convert to relative coordinates
    base_x, base_y = 0, 0
    for index, point in enumerate(temp_point_history):
        if index == 0:
            base_x, base_y = point[0], point[1]

        temp_point_history[index][0] = (temp_point_history[index][0] -
                                        base_x) / image_width
        temp_point_history[index][1] = (temp_point_history[index][1] -
                                        base_y) / image_height

    # Convert to a one-dimensional list
    temp_point_history = list(
        itertools.chain.from_iterable(temp_point_history))

    return temp_point_history

def logging_csv(number, mode, landmark_list, point_history_list):
    # Your CSV logging function here
    if mode == 0:
        pass
    if mode == 1 and (0 <= number <= 9):
        csv_path = 'model/keypoint_classifier/keypoint.csv'
        with open(csv_path, 'a', newline="") as f:
            writer = csv.writer(f)
            writer.writerow([number, *landmark_list])
    if mode == 2 and (0 <= number <= 9):
        csv_path = 'model/point_history_classifier/point_history.csv'
        
        with open(csv_path, 'a', newline="") as f:
            writer = csv.writer(f)
            writer.writerow([number, *point_history_list])
    return

def draw_landmarks(image, landmark_point):
    # Your landmark drawing function here
    if len(landmark_point) > 0:
        # Thumb
        cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[3]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[3]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[3]), tuple(landmark_point[4]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[3]), tuple(landmark_point[4]),
                (255, 255, 255), 2)

        # Index finger
        cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[6]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[6]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[6]), tuple(landmark_point[7]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[6]), tuple(landmark_point[7]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[7]), tuple(landmark_point[8]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[7]), tuple(landmark_point[8]),
                (255, 255, 255), 2)

        # Middle finger
        cv.line(image, tuple(landmark_point[9]), tuple(landmark_point[10]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[9]), tuple(landmark_point[10]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[10]), tuple(landmark_point[11]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[10]), tuple(landmark_point[11]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[11]), tuple(landmark_point[12]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[11]), tuple(landmark_point[12]),
                (255, 255, 255), 2)

        # Ring finger
        cv.line(image, tuple(landmark_point[13]), tuple(landmark_point[14]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[13]), tuple(landmark_point[14]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[14]), tuple(landmark_point[15]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[14]), tuple(landmark_point[15]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[15]), tuple(landmark_point[16]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[15]), tuple(landmark_point[16]),
                (255, 255, 255), 2)

        # Little finger
        cv.line(image, tuple(landmark_point[17]), tuple(landmark_point[18]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[17]), tuple(landmark_point[18]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[18]), tuple(landmark_point[19]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[18]), tuple(landmark_point[19]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[19]), tuple(landmark_point[20]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[19]), tuple(landmark_point[20]),
                (255, 255, 255), 2)

        # Palm
        cv.line(image, tuple(landmark_point[0]), tuple(landmark_point[1]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[0]), tuple(landmark_point[1]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[1]), tuple(landmark_point[2]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[1]), tuple(landmark_point[2]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[5]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[5]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[9]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[9]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[9]), tuple(landmark_point[13]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[9]), tuple(landmark_point[13]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[13]), tuple(landmark_point[17]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[13]), tuple(landmark_point[17]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[17]), tuple(landmark_point[0]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[17]), tuple(landmark_point[0]),
                (255, 255, 255), 2)

    # Key Points
    for index, landmark in enumerate(landmark_point):
        if index == 0:  
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 1:  
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 2:  
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 3:  
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 4:  
            cv.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)
        if index == 5:  
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 6:  
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 7:  
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 8:  
            cv.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)
        if index == 9:  
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 10:  
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 11:  
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 12:  
            cv.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)
        if index == 13:  
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 14:  
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 15:  
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 16:  
            cv.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)
        if index == 17:  
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 18:  
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 19:  
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 20:  
            cv.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)

    return image

def draw_bounding_rect(use_brect, image, brect):
    # Your bounding rect drawing function here
    if use_brect:
        # Outer rectangle
        cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[3]),
                     (0, 0, 0), 1)

    return image

def draw_info_text(image, brect, handedness, hand_sign_text, finger_gesture_text):
    # Your info text drawing function here
    cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[1] - 22),
                 (0, 0, 0), -1)

    info_text = handedness.classification[0].label[0:]
    if hand_sign_text != "":
        info_text = info_text + ':' + hand_sign_text
    cv.putText(image, info_text, (brect[0] + 5, brect[1] - 4),
               cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv.LINE_AA)

    if finger_gesture_text != "":
        cv.putText(image, "Finger Gesture:" + finger_gesture_text, (10, 60),
                   cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 4, cv.LINE_AA)
        cv.putText(image, "Finger Gesture:" + finger_gesture_text, (10, 60),
                   cv.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2,
                   cv.LINE_AA)

    return image

def draw_point_history(image, point_history):
    # Your point history drawing function here
    for index, point in enumerate(point_history):
        if point[0] != 0 and point[1] != 0:
            cv.circle(image, (point[0], point[1]), 1 + int(index / 2),
                      (152, 251, 152), 2)

    return image

def draw_info(image, fps, mode, number):
    # Your info drawing function here
    cv.putText(image, "FPS:" + str(fps), (10, 30), cv.FONT_HERSHEY_SIMPLEX,
               1.0, (0, 0, 0), 4, cv.LINE_AA)
    cv.putText(image, "FPS:" + str(fps), (10, 30), cv.FONT_HERSHEY_SIMPLEX,
               1.0, (255, 255, 255), 2, cv.LINE_AA)

    mode_string = ['Logging Key Point', 'Logging Point History']
    if 1 <= mode <= 2:
        cv.putText(image, "MODE:" + mode_string[mode - 1], (10, 90),
                   cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1,
                   cv.LINE_AA)
        if 0 <= number <= 9:
            cv.putText(image, "NUM:" + str(number), (10, 110),
                       cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1,
                       cv.LINE_AA)
    return image

# # Video streaming generator
# def generate_frames():
#     cap = cv.VideoCapture(args["device"])
#     cap.set(cv.CAP_PROP_FRAME_WIDTH, args["width"])
#     cap.set(cv.CAP_PROP_FRAME_HEIGHT, args["height"])
#     cvFpsCalc = CvFpsCalc(buffer_len=10)

#     mode = 0
#     use_brect = True

#     while True:
#         fps = cvFpsCalc.get()

#         ret, image = cap.read()
#         if not ret:
#             break

#         image = cv.flip(image, 1)
#         debug_image = copy.deepcopy(image)

#         # Process hand detection
#         image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
#         image.flags.writeable = False
#         results = hands.process(image)
#         image.flags.writeable = True

#         if results.multi_hand_landmarks is not None:
#             for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
#                 brect = calc_bounding_rect(debug_image, hand_landmarks)
#                 landmark_list = calc_landmark_list(debug_image, hand_landmarks)

#                 # Process landmark data
#                 pre_processed_landmark_list = pre_process_landmark(landmark_list)
#                 pre_processed_point_history_list = pre_process_point_history(debug_image, point_history)

#                 # Hand sign classification
#                 hand_sign_id = keypoint_classifier(pre_processed_landmark_list)
#                 if hand_sign_id == 2:  # Point gesture
#                     point_history.append(landmark_list[8])
#                 else:
#                     point_history.append([0, 0])

#                 # Finger gesture classification
#                 finger_gesture_id = 0
#                 if len(pre_processed_point_history_list) == (history_length * 2):
#                     finger_gesture_id = point_history_classifier(pre_processed_point_history_list)

#                 finger_gesture_history.append(finger_gesture_id)
#                 most_common_fg_id = Counter(finger_gesture_history).most_common()

#                 debug_image = draw_bounding_rect(use_brect, debug_image, brect)
#                 debug_image = draw_landmarks(debug_image, landmark_list)
#                 debug_image = draw_info_text(
#                     debug_image, brect, handedness,
#                     keypoint_classifier_labels[hand_sign_id],
#                     point_history_classifier_labels[most_common_fg_id[0][0]],
#                 )

#         debug_image = draw_point_history(debug_image, point_history)
#         debug_image = draw_info(debug_image, fps, mode, 0)

#         # Encode the frame
#         ret, buffer = cv.imencode(".jpg", debug_image)
#         frame = buffer.tobytes()
#         yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")

#     cap.release()
recognized_words = []

def load_gesture_labels(file_path):
    with open(file_path, 'r') as f:
        gesture_labels = [line.strip() for line in f]  # Read lines into a list
    return gesture_labels

# ########## LABELS  L O A D I N G  USING  M O D E L
gesture_labels = load_gesture_labels("model/keypoint_classifier/keypoint_classifier_label.csv")

def generate_frames():
    global recognized_words
    cap = cv.VideoCapture(args["device"])
    cap.set(cv.CAP_PROP_FRAME_WIDTH, args["width"])
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, args["height"])
    cvFpsCalc = CvFpsCalc(buffer_len=10)

    mode = 0
    use_brect = True

    while True:
        fps = cvFpsCalc.get()

        ret, image = cap.read()
        if not ret:
            break

        image = cv.flip(image, 1)
        debug_image = copy.deepcopy(image)

        # Process hand detection
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = hands.process(image)
        image.flags.writeable = True

        if results.multi_hand_landmarks is not None:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                brect = calc_bounding_rect(debug_image, hand_landmarks)
                landmark_list = calc_landmark_list(debug_image, hand_landmarks)

                # Process landmark data
                pre_processed_landmark_list = pre_process_landmark(landmark_list)

                # Hand sign classification
                hand_sign_index = keypoint_classifier(pre_processed_landmark_list)
                if 0 <= hand_sign_index < len(gesture_labels):
                    word = gesture_labels[hand_sign_index]
                else:
                    word = "Unknown Gesture"  # Fallback for unexpected index

                # Add recognized word to the list
                if word:
                    recognized_words.append(word)
                    # Limit the number of words to avoid overflow
                    if len(recognized_words) > 50:
                        recognized_words.pop(0)

        debug_image = draw_point_history(debug_image, point_history)
        debug_image = draw_info(debug_image, fps, mode, 0)

        # Encode the frame
        ret, buffer = cv.imencode(".jpg", debug_image)
        frame = buffer.tobytes()
        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")

    cap.release()

@app.route("/get_words")
def get_words():
    global recognized_words
    return {"words": recognized_words}

@app.route("/clear_words", methods=["POST"])
def clear_words():
    global recognized_words
    recognized_words = []  # Clear the list
    print("Recognized words cleared")  # Debugging
    return "", 204  # Respond with no content

@app.route("/")
def openning():
    return render_template("Openning.html")

@app.route("/opening1.html", methods=["GET", "POST"])
def index():
    global args
    if request.method == "POST":
        args["device"] = int(request.form.get("device", 0))
        args["width"] = int(request.form.get("width", 960))
        args["height"] = int(request.form.get("height", 540))
        args["min_detection_confidence"] = float(request.form.get("min_detection_confidence", 0.7))
        args["min_tracking_confidence"] = float(request.form.get("min_tracking_confidence", 0.5))
        initialize_models()
    return render_template("opening1.html", args=args)

@app.route("/about_us")
def about():
    return render_template("about_us.html")

@app.route('/dashboard')
def dashboard():
    return render_template("dashboard.html")

@app.route('/reg', methods=['GET', 'POST'])
def registration():
    if request.method == "POST":
        # Get form data
        first_name = request.form['first_name']
        last_name = request.form['last_name']
        email = request.form['email']
        dob = request.form['dob']
        password = request.form['password']

        # Hash password
        hashed_pw = generate_password_hash(password)

        # Insert into DB
        cursor = mysql.connection.cursor()
        cursor.execute(
            "INSERT INTO users (first_name, last_name, email, dob, password_hash) VALUES (%s, %s, %s, %s, %s)",
            (first_name, last_name, email, dob, hashed_pw)
        )
        mysql.connection.commit()
        cursor.close()

        flash("User Registered Successfully!")
        return redirect('/dashboard')
    
    # Render registration form on GET
    return render_template("reg.html")


# Login
@app.route('/log', methods=['GET', 'POST']) 
def login():
    if request.method == "POST":
        email = request.form.get('email')
        password = request.form.get('password')

        if not email or not password:
            flash("Email and password are required.")
            return redirect('/dashboard')

        cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        cursor.execute("SELECT id, email, password_hash FROM users WHERE email = %s", (email,))
        user = cursor.fetchone()
        cursor.close()

        if user:
            user_id = user['id']
            stored_email = user['email']
            stored_hash = user['password_hash']

            print("Entered Password:", password)
            print("Stored Hash from DB:", stored_hash)
            print("Hash Match?", check_password_hash(stored_hash, password))

            if check_password_hash(stored_hash, password):
                session['user_id'] = user_id
                flash("Successfully login")
                return redirect('/dashboard')
            else:
                flash("Invalid credentials.")
                return redirect('/dashboard')
        else:
            flash("Invalid credentials.")
            return redirect('/dashboard')

    return render_template("log.html")



@app.route("/video_feed")
def video_feed():
    return Response(generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")


def open_browser():
      webbrowser.open_new("http://127.0.0.1:5000/")

if __name__ == "__main__":
    Timer(1, open_browser).start()
    # app.run(port=2000)
    app.run(debug=True, host="0.0.0.0", port=5000)

