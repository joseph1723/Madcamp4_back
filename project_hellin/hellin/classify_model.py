import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
import cv2
import time
import random
import numpy as np 
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from collections import deque
from keras.models import load_model
# from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, f1_score

def classify_video(video_path):
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(BASE_DIR, 'classify_workout_model.tflite')
    # workout_model = load_model(model_path)
    # 1/0
# Convert the model to TensorFlow Lite format
    # converter = tf.lite.TFLiteConverter.from_keras_model(workout_model)
    # tflite_model = converter.convert()
    
    # with open('classify_workout_model.tflite', 'wb') as f:
        # f.write(tflite_model)
    # print("CHECKOUT")

    interpreter = tf.lite.Interpreter(model_path = model_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # print("== Input details ==")
    # print("name:", input_details[0]['name'])
    # print("shape:", input_details[0]['shape'])
    # print("type:", input_details[0]['dtype'])

    # print("\n== Output details ==")
    # print("name:", output_details[0]['name'])
    # print("shape:", output_details[0]['shape'])
    # print("type:", output_details[0]['dtype'])

    labels = []
    with open(f'{BASE_DIR}/workout_label.txt', 'r') as f:
        for row in f:
            labels.append(row)

    video_capture = cv2.VideoCapture(video_path)
    # writer = None
    H, W = None, None
    Q = deque(maxlen=128)
    n = 0
    img_size = (input_details[0]['shape'][1], input_details[0]['shape'][2])

    fps = video_capture.get(cv2.CAP_PROP_FPS)
    frame_count = video_capture.get(cv2.CAP_PROP_FRAME_COUNT)
    duration = frame_count/fps

    # print(f'Duration: {duration}')
    # print(f'FPS: {fps}')
    # print(f'Total frame: {frame_count}\n')

    # print('Prediction process')
    start_time = time.time()

    scan_count = 0
    # Loop through each frame in the video
    while True:
        # read a frame
        success, frame = video_capture.read()
        
        # if frame not read properly then break the loop
        if not success:
            break
        
        # count the frame
        n += 1
        
        if duration < 5.0:
            # predict every 10 frame (1, 11, 21, ... etc)
            step = 10
        elif duration < 10.0:
            # predict every 15 frame (1, 16, 31, ... etc)
            step = 15
        else:
            # predict every 30 frame (1, 31, 61, ... etc)
            step = 30
        
        if n % step != 1:
            continue
        
        # get frame dimensions
        if W is None or H is None:
            (H, W) = frame.shape[:2]
        
        # clone the frame for the output then preprocess the frame for the prediction
        output = frame.copy()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, img_size).astype("float32")
        
        interpreter.set_tensor(input_details[0]['index'], np.expand_dims(frame, axis=0))

        interpreter.invoke()

        predictions = interpreter.get_tensor(output_details[0]['index'])[0]
        
        Q.append(predictions)
        
        results = np.array(Q).mean(axis=0)
        i = np.argmax(results)
        label = labels[i]
        
        text = f'{label}'
    #     cv2.putText(output, text, (35, 50), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), 10)
        
    #     if writer is None:
    #         fourcc = cv2.VideoWriter_fourcc(*"XVID")
    #         writer = cv2.VideoWriter('output.avi', fourcc, 30, (W, H), True)
            
    #     writer.write(output)
        
    #     plt.imshow(output)
    #     plt.axis('off')
        
        # print(f'{n}, {text}\t{results[i]}')
        scan_count += 1
        
        target_frame = int(fps*2)
        
        # break the loop if prediction > 90% and video already more than 2 seconds (fps*2)
        if results[i] >= 0.9 and n >= target_frame:
            break

    end_time = time.time()

    # print(f'\nActual video: {random_file}')
    # print(f'Prediction: {text}')
    # print(f'confidence: {results[i]}')
    # print(f'Prediction time: {end_time - start_time} sec')
    # print(f'Scan speed: {(end_time - start_time)/scan_count} per frame\n')

    # writer.release()
    # video_capture.release()

    result_df = pd.DataFrame({'exercise': labels,
                            'percentage': results
                            })
    print(labels, results)
    print(labels[np.argmax(np.array(results))].split(','))
    return labels[np.argmax(np.array(results))].split(',')

classify_video('/root/madcamp4_back/project_hellin/media/classifytest.mp4')