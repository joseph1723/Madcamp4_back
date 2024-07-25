from rest_framework.decorators import api_view
from rest_framework.response import Response
from django.views.decorators.csrf import csrf_exempt
import json
from django.http import JsonResponse

@csrf_exempt
def example_view(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            name = data.get('name')
            # 여기서 데이터를 처리합니다.
            response_data = {'message': f'Received name: {name}'}
            return JsonResponse(response_data, status=200)
        except json.JSONDecodeError:
            return JsonResponse({'error': 'Invalid JSON'}, status=400)
    else:
        return JsonResponse({'error': 'Invalid HTTP method'}, status=405)

import os
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.core.files.storage import default_storage
from .squat_predict import process_video
from .classify_model import classify_video
@csrf_exempt
def upload_view(request):
    if request.method == 'POST':
        if 'file' in request.FILES and 'exercise_type' in request.POST:
            uploaded_file = request.FILES['file']
            exercise_type = request.POST['exercise_type']

            file_path = default_storage.save(uploaded_file.name, uploaded_file)
            video_path = os.path.join(default_storage.location, file_path)
            
            # Output file path
            # output_file = os.path.join(default_storage.location, 'feedback.txt')
            print(video_path, file_path)
            # Call the process_video function
            comment = process_video(video_path, exercise_type)
            if os.path.exists(video_path):
                os.remove(video_path)
            return JsonResponse({'message': 'File uploaded successfully', 'file_path': file_path, 'advice': comment})

        return JsonResponse({'error': 'No file uploaded'}, status=400)
    
    return JsonResponse({'error': 'Invalid request method'}, status=405)
@csrf_exempt
def classify_view(request):
    if request.method == 'POST':
        print("ERER")
        print(request.FILES)
        if 'file' in request.FILES:
            uploaded_file = request.FILES['file']
            print("HERE")

            file_path = default_storage.save(uploaded_file.name, uploaded_file)
            video_path = os.path.join(default_storage.location, file_path)
            
            print(video_path, file_path)
            # Call the process_video function
            predict = classify_video(video_path)
            if os.path.exists(video_path):
                os.remove(video_path)
            return JsonResponse({'message': 'File uploaded successfully', 'predict': predict[0], 'url':predict[1]})
        return JsonResponse({'error': 'No file uploaded', 'message':"No file uploaded to server"}, status=400)
    return JsonResponse({'error': 'Invalid request method'}, status=405)