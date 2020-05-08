from django.shortcuts import render
from django.http import HttpResponse
import cv2
from .code.process_video import get_features_frames, score_video_log_2
from .code.video_utils import CatVideo
import numpy as np
import os
from threading import Thread
from django.http import JsonResponse


def index(request):
    return render(request, 'apa/index.html')


def process(request):
    if request.method == "POST":
        file = request.FILES['filename']
        key = request.session._get_or_create_session_key()
        video_file = load_video(file, key)
        dest_file = f"apa/media/image_{key}.jpg"
        frames = list(CatVideo(video_file).iter_all_frames())
        thread = Thread(target=run_model, args=(frames, dest_file))
        thread.start()
        response = JsonResponse(data={"key": key})
        return response
    elif request.method == "GET":
        key = request.GET["key"]
        if os.path.exists(f"apa/media/image_{key}.jpg"):
            os.remove(f"apa/media/video_{key}.mp4")
            return JsonResponse(data={"success": 1})
        else:
            return JsonResponse(data={"success": 0})


def result(request):
    if request.method == "GET":
        key = request.GET["key"]
        opt_img_path = f"media/image_{key}.jpg"
        return render(request, 'apa/result.html', {"opt_img": opt_img_path})


# Helper functions
def load_video(file, key):
    dest_file = f'apa/media/video_{key}.mp4'
    with open(dest_file, 'wb+') as dest:
        for chunk in file.chunks():
            dest.write(chunk)
    return dest_file


def run_model(frames, dest_file):
    features = get_features_frames(frames)
    opt_img = frames[score_video_log_2(features)]
    cv2.imwrite(dest_file, opt_img[:, :, ::-1])
