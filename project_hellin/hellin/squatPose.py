import cv2
import mediapipe as mp
import math
import numpy as np

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

first_frame_left_heel = {}

EXERCISES = [
    'squats',
    'planks',
]

def get_angle(v1, v2):
    dot = np.dot(v1, v2)
    mod_v1 = np.linalg.norm(v1)
    mod_v2 = np.linalg.norm(v2)
    cos_theta = dot/(mod_v1*mod_v2)
    theta = math.acos(cos_theta)
    return theta


def get_length(v):
    return np.dot(v, v)**0.5



def vector(p1, p2):
    return np.array([p2[i] - p1[i] for i in range(len(p1))])

def unit_vector(vector):
    return vector / np.linalg.norm(vector)

def angle_between(v1, v2):
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    dot_product = np.dot(v1_u, v2_u)
    return np.arccos(np.clip(dot_product, -1.0, 1.0))

def get_params(results, base_value):
    
# 예시 사용법
    labels = ["NOSE", "LEFT_EYE", "RIGHT_EYE","LEFT_EAR", "RIGHT_EAR", "LEFT_SHOULDER", "RIGHT_SHOULDER", "LEFT_ELBOW", "RIGHT_ELBOW",
          "LEFT_WRIST", "RIGHT_WRIST", "LEFT_HIP", "RIGHT_HIP", "LEFT_KNEE",
          "RIGHT_KNEE", "LEFT_ANKLE", "RIGHT_ANKLE", "LEFT_FOOT_INDEX", "RIGHT_FOOT_INDEX" , "LEFT_HEEL", "RIGHT_HEEL",
          'LEFT_ANKLE_ANGLE','RIGHT_ANKLE_ANGLE', 'LEFT_KNEE_ANGLE','RIGHT_KNEE_ANGLE','LEFT_HIP_ANGLE',
          'RIGHT_HIP_ANGLE','LEFT_GROIN_ANGLE','RIGHT_GROIN_ANGLE','LEFT_SHOULDER_ANGLE','RIGHT_SHOULDER_ANGLE','LEFT_ELBOW_ANGLE','RIGHT_ELBOW_ANGLE'
          ]
    if results.pose_landmarks is None:
        # 모든 포인트가 결측값인 경우, labels의 길이만큼 0으로 채운 배열 반환
        return np.zeros(len(labels) * 3)

    # 모든 포인트를 딕셔너리에 저장
    points = {}
    landmarks = [
        mp_pose.PoseLandmark.NOSE,
        mp_pose.PoseLandmark.LEFT_EYE_INNER,
        mp_pose.PoseLandmark.LEFT_EYE,
        mp_pose.PoseLandmark.LEFT_EYE_OUTER,
        mp_pose.PoseLandmark.RIGHT_EYE_INNER,
        mp_pose.PoseLandmark.RIGHT_EYE,
        mp_pose.PoseLandmark.RIGHT_EYE_OUTER,
        mp_pose.PoseLandmark.LEFT_EAR,
        mp_pose.PoseLandmark.RIGHT_EAR,
        mp_pose.PoseLandmark.LEFT_SHOULDER,
        mp_pose.PoseLandmark.RIGHT_SHOULDER,
        mp_pose.PoseLandmark.LEFT_ELBOW,
        mp_pose.PoseLandmark.RIGHT_ELBOW,
        mp_pose.PoseLandmark.LEFT_WRIST,
        mp_pose.PoseLandmark.RIGHT_WRIST,
        mp_pose.PoseLandmark.LEFT_PINKY,
        mp_pose.PoseLandmark.RIGHT_PINKY,
        mp_pose.PoseLandmark.LEFT_INDEX,
        mp_pose.PoseLandmark.RIGHT_INDEX,
        mp_pose.PoseLandmark.LEFT_THUMB,
        mp_pose.PoseLandmark.RIGHT_THUMB,
        mp_pose.PoseLandmark.LEFT_HIP,
        mp_pose.PoseLandmark.RIGHT_HIP,
        mp_pose.PoseLandmark.LEFT_KNEE,
        mp_pose.PoseLandmark.RIGHT_KNEE,
        mp_pose.PoseLandmark.LEFT_ANKLE,
        mp_pose.PoseLandmark.RIGHT_ANKLE,
        mp_pose.PoseLandmark.LEFT_HEEL,
        mp_pose.PoseLandmark.RIGHT_HEEL,
        mp_pose.PoseLandmark.LEFT_FOOT_INDEX,
        mp_pose.PoseLandmark.RIGHT_FOOT_INDEX
    ]


    for landmark in landmarks:
        name = landmark.name
        if name in labels:
            landmark_index = mp_pose.PoseLandmark[landmark.name].value
            pose_landmark = results.pose_landmarks.landmark[landmark_index]
            points[name] = np.array([pose_landmark.x, pose_landmark.y, pose_landmark.z])
    calculate_angles(points)
    # labels 목록에서 필요한 포인트만 선택하고 평탄화
    flat_params = []
    for label in labels:
        if label in points:
            flat_params.extend(points[label])
        else:
            # 없는 label에 대해서는 0으로 채운다.
            assert False, f"{label} is not in mediapipe"
            flat_params.extend([0, 0, 0])
    # np.array로 변환하고 1차원 배열로 반환
    return np.array(flat_params)

def calculate_angles(row):
    angles = {
        'LEFT_ANKLE': [],
        'RIGHT_ANKLE': [],
        'LEFT_KNEE': [],
        'RIGHT_KNEE': [],
        'LEFT_HIP': [],
        'RIGHT_HIP': [],
        'LEFT_GROIN': [],
        'RIGHT_GROIN': [],
        'LEFT_SHOULDER': [],
        'RIGHT_SHOULDER': [],
        'LEFT_ELBOW': [],
        'RIGHT_ELBOW': [],
    }
    
        # 좌표 가져오기
    LEFT_ANKLE = (row['LEFT_ANKLE'])
    RIGHT_ANKLE = (row['RIGHT_ANKLE'])
    LEFT_FOOT = (row['LEFT_FOOT_INDEX'])
    RIGHT_FOOT = (row['RIGHT_FOOT_INDEX'])
    LEFT_KNEE = (row['LEFT_KNEE'])
    RIGHT_KNEE = (row['RIGHT_KNEE'])
    LEFT_HIP = (row['LEFT_HIP'])
    RIGHT_HIP = (row['RIGHT_HIP'])
    LEFT_SHOULDER = (row['LEFT_SHOULDER'])
    RIGHT_SHOULDER = (row['RIGHT_SHOULDER'])
    LEFT_ELBOW = (row['LEFT_ELBOW'])
    RIGHT_ELBOW = (row['RIGHT_ELBOW'])
    LEFT_WRIST = (row['LEFT_WRIST'])
    RIGHT_WRIST = (row['RIGHT_WRIST'])



        # 발목 각도
    vector1 = vector(LEFT_ANKLE, LEFT_KNEE)
    vector2 = vector(LEFT_ANKLE, LEFT_FOOT)
    angles['LEFT_ANKLE'].append(angle_between(vector1, vector2))
    vector1 = vector(RIGHT_ANKLE, RIGHT_KNEE)
    vector2 = vector(RIGHT_ANKLE, RIGHT_FOOT)
    angles['RIGHT_ANKLE'].append(angle_between(vector1, vector2))

    # 무릎 각도
    vector1 = vector(LEFT_KNEE, LEFT_HIP)
    vector2 = vector(LEFT_KNEE, LEFT_ANKLE)
    angles['LEFT_KNEE'].append(angle_between(vector1, vector2))
    vector1 = vector(RIGHT_KNEE, RIGHT_HIP)
    vector2 = vector(RIGHT_KNEE, RIGHT_ANKLE)
    angles['RIGHT_KNEE'].append(angle_between(vector1, vector2))

    # 고관절 각도
    vector1 = vector(LEFT_HIP, LEFT_SHOULDER)
    vector2 = vector(LEFT_HIP, LEFT_KNEE)
    angles['LEFT_HIP'].append(angle_between(vector1, vector2))
    vector1 = vector(RIGHT_HIP, RIGHT_SHOULDER)
    vector2 = vector(RIGHT_HIP, RIGHT_KNEE)
    angles['RIGHT_HIP'].append(angle_between(vector1, vector2))

    # 사타구니 사이 각도
    vector1 = vector(LEFT_HIP, RIGHT_HIP)
    vector2 = vector(LEFT_HIP, LEFT_KNEE)
    angles['LEFT_GROIN'].append(angle_between(vector1, vector2))
    vector1 = vector(RIGHT_HIP, LEFT_HIP)
    vector2 = vector(RIGHT_HIP, RIGHT_KNEE)
    angles['RIGHT_GROIN'].append(angle_between(vector1, vector2))

    # 어깨 각도
    vector1 = vector(LEFT_SHOULDER, LEFT_ELBOW)
    vector2 = vector(LEFT_SHOULDER, LEFT_HIP)
    angles['LEFT_SHOULDER'].append(angle_between(vector1, vector2))
    vector1 = vector(RIGHT_SHOULDER, RIGHT_ELBOW)
    vector2 = vector(RIGHT_SHOULDER, RIGHT_HIP)
    angles['RIGHT_SHOULDER'].append(angle_between(vector1, vector2))

    # 팔꿈치 각도
    vector1 = vector(LEFT_ELBOW, LEFT_WRIST)
    vector2 = vector(LEFT_ELBOW, LEFT_SHOULDER)
    angles['LEFT_ELBOW'].append(angle_between(vector1, vector2))
    vector1 = vector(RIGHT_ELBOW, RIGHT_WRIST)
    vector2 = vector(RIGHT_ELBOW, RIGHT_SHOULDER)
    angles['RIGHT_ELBOW'].append(angle_between(vector1, vector2))

    
    row['LEFT_ANKLE_ANGLE'] = angles['LEFT_ANKLE']
    row['RIGHT_ANKLE_ANGLE'] = angles['RIGHT_ANKLE']
    row['LEFT_KNEE_ANGLE'] = angles['LEFT_KNEE']
    row['RIGHT_KNEE_ANGLE'] = angles['RIGHT_KNEE']
    row['LEFT_HIP_ANGLE'] = angles['LEFT_HIP']
    row['RIGHT_HIP_ANGLE'] = angles['RIGHT_HIP']
    row['LEFT_GROIN_ANGLE'] = angles['LEFT_GROIN']
    row['RIGHT_GROIN_ANGLE'] = angles['RIGHT_GROIN']
    row['LEFT_SHOULDER_ANGLE'] = angles['LEFT_SHOULDER']
    row['RIGHT_SHOULDER_ANGLE'] = angles['RIGHT_SHOULDER']
    row['LEFT_ELBOW_ANGLE'] = angles['LEFT_ELBOW']
    row['RIGHT_ELBOW_ANGLE'] = angles['RIGHT_ELBOW']


