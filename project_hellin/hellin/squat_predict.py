import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_dangerously_set_inner_html
import mediapipe as mp
from . import squatPose as sp
from flask import Flask, Response
import cv2
import tensorflow as tf
import numpy as np
# from utils import landmarks_list_to_array, label_params, label_final_results
from tensorflow.python.framework import ops
import os
import requests
import google.generativeai as genai

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
genai.configure(api_key='AIzaSyB7y9Bji5w_rlYPkn6bdwbt83kKMjK7yvw')
genai_model = genai.GenerativeModel('gemini-1.5-flash')

# model = tf.keras.models.load_model("./airsquat_working_model_3.0.0")
def load_model(exercise):

    # Define the path to your model file
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(BASE_DIR, f'{exercise}_working_model')

    # Check if the file exists
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")

    # Load the model
    model = tf.keras.models.load_model(model_path)
    return model

def process_video(video_path, exercise_type):
    model = load_model(exercise_type)
    print(f"process_video called {video_path}")
    cap = cv2.VideoCapture(video_path)
    j = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    feedback = []
    inputs = []
    with mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                break

            image_height, image_width, _ = image.shape
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            dim = (image_width // 5, image_height // 5)
            resized_image = cv2.resize(image, dim)
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            results = pose.process(resized_image)

            if not j:
                base_value = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HEEL]
            # if exercise_type == 'airsquat':
            params = sp.get_params(results, base_value)
            # flat_params = np.reshape(params, (5, 1))
            p = np.insert(params, 0, j/total_frames*100)
            # print(p)
            # output = model.predict([j]+params)
            inputs.append(p)

            j += 1
            print(f"Frame {j} is executed")
    inputs = np.array(inputs)
    # print(model.summary())
    # print(inputs.shape)
    print(inputs)
    print(inputs.shape)
    print(exercise_type)
    output = model.predict(inputs)
    print(output)
    comments = comment(exercise_type, list(output), total_frames)
    cap.release()

    return comments

if __name__ == '__main__':
    video_path = '../media/squat_hyeono_front_good.mp4'  # 비디오 파일 경로를 설정하세요.
    output_file = 'feedback.txt'  # 출력 파일 경로를 설정하세요.
    process_video(video_path, output_file)
    print(f"Feedback written to {output_file}")

def comment(exercise, output, total_frame):
    comments = ""
    votes = [[0,0,0,0], [0,0,0,0], [0,0,0,0]]
    if exercise == 'airsquat':
        critation = [[0.3, 0.5, 0.3, 0.2],[0.31, 0.5, 0.3, 0.2],[0.15, 0.35, 0.5, 0.2]]
        error = ['이상적인 자세', '고관절이상', '발의 위치 이상', '엉덩이하방 오류']
        for idx, row in enumerate(output):
            if idx < total_frame / 3:
                i = 0
            elif idx <  2.2 * total_frame / 3:
                i = 1
            else:
                i = 2
            votes[i] = [v + (o > c) for v, (o, c) in zip(votes[i], zip(row, critation[i]))]
            
    elif exercise == 'deadlift':
        critation = [[0.3, 0.5, 0.3, 0.2],[0.31, 0.5, 0.3, 0.2],[0.15, 0.35, 0.5, 0.2]]
        error = ['가슴및고관절오류', '이상적인자세', '팔자세오류', '고관절이동오류']
        for idx, row in enumerate(output):
            if idx < total_frame / 3:
                i = 0
            elif idx <  2.2 * total_frame / 3:
                i = 1
            else:
                i = 2
            votes[i] = [v + (o > c) for v, (o, c) in zip(votes[i], zip(row, critation[i]))]
    elif exercise == 'shoulderpress':
        critation = [[0.3, 0.5, 0.3, 0.2],[0.31, 0.5, 0.3, 0.2],[0.15, 0.35, 0.5, 0.2]]
        error = ['팔움직임오류','이상적인 자세', '팔꿈치및어깨비활성화오류', '오버헤드팔중심오류']
        for idx, row in enumerate(output):
            if idx < total_frame / 3:
                i = 0
            elif idx <  2.2 * total_frame / 3:
                i = 1
            else:
                i = 2
            votes[i] = [v + (o > c) for v, (o, c) in zip(votes[i], zip(row, critation[i]))]
    elif exercise == 'pushpress':
        critation = [[0.3, 0.5, 0.3, 0.2],[0.31, 0.5, 0.3, 0.2],[0.15, 0.35, 0.5, 0.2]]
        error = ['이상적인 자세', '고관절신장오류', '고관절쏠림오류', '가슴기울임오류']
        for idx, row in enumerate(output):
            if idx < total_frame / 3:
                i = 0
            elif idx <  2.2 * total_frame / 3:
                i = 1
            else:
                i = 2
            votes[i] = [v + (o > c) for v, (o, c) in zip(votes[i], zip(row, critation[i]))]
    elif exercise == 'overheadsquat':
        critation = [[0.3, 0.5, 0.3],[0.31, 0.5, 0.3],[0.15, 0.35, 0.5]]
        error = ['정상','오버헤드팔중심오류','팔꿈치비활성화오류']
        for idx, row in enumerate(output):
            if idx < total_frame / 3:
                i = 0
            elif idx <  2.2 * total_frame / 3:
                i = 1
            else:
                i = 2
            votes[i] = [v + (o > c) for v, (o, c) in zip(votes[i], zip(row, critation[i]))]
    elif exercise == 'sumodeadlifthighpull':
        critation = [[0.3, 0.5, 0.3],[0.31, 0.5, 0.3],[0.15, 0.35, 0.5]]
        error = ['정상','오버헤드팔중심오류','팔꿈치비활성화오류']
        for idx, row in enumerate(output):
            if idx < total_frame / 3:
                i = 0
            elif idx <  2.2 * total_frame / 3:
                i = 1
            else:
                i = 2
            votes[i] = [v + (o > c) for v, (o, c) in zip(votes[i], zip(row, critation[i]))]
    elif exercise == 'medicineball':
        critation = [[0.3, 0.5, 0.3, 0.2],[0.31, 0.5, 0.3, 0.2],[0.15, 0.35, 0.5, 0.2]]
        error = ['팔동작오류','정상','풀동작오류','고관절신전부재오류']
        for idx, row in enumerate(output):
            if idx < total_frame / 3:
                i = 0
            elif idx <  2.2 * total_frame / 3:
                i = 1
            else:
                i = 2
            votes[i] = [v + (o > c) for v, (o, c) in zip(votes[i], zip(row, critation[i]))]
    elif exercise == 'frontsquat':
        critation = [[0.3, 0.5, 0.3],[0.31, 0.5, 0.3],[0.15, 0.35, 0.5]]
        error = ['정상','팔꿈치오류','팔꿈치하방오류']
        for idx, row in enumerate(output):
            if idx < total_frame / 3:
                i = 0
            elif idx <  2.2 * total_frame / 3:
                i = 1
            else:
                i = 2
            votes[i] = [v + (o > c) for v, (o, c) in zip(votes[i], zip(row, critation[i]))]
    else:
        return f"ERROR- {exercise} IS NOT A WORK OUT WE SERVE"
    api_data = {
        "exercise": exercise,
        "votes": votes,
        "errors": error
    }
    # Call Geminai API
    prompt = f"The following list is a parameter representing how user is doing good. analysis[0] is ealier part, analysis[1] is middle part, and analysis[2] is later part. each analysis[i] has their own paramters, and they mean an advice which is same as a list named error. error is {error}, analysis is {votes}. Make a good advice to the user, as if you're their personal trainer. Do not say exact values of parameters. Translate your advice into Korean"
    
    prompt = f'''These are the list of parameter representing the performance of user in workout. {votes[0]} indicates earlier part of the performance, {votes[1]} indicates the middle part, and {votes[2]} indicates the end. Numbers given in each list refers to list-named errors - {error}, and analysis as {votes}.
                Print out brief essential advice for each part, as if you're a kind, professional personal trainer of the user - in Korean - excluding the numeric analysis of users' performance.'''
    response = genai_model.generate_content(prompt)
    # if response.status_code == 200:
        # Assuming the API returns the comments as a JSON response
        # api_response = response.json()
        # print(response.text)
    # else:
        # return
        # comments = "Failed to get comments from Geminai API."
    print(exercise)
    print(response.text)
    return response.text