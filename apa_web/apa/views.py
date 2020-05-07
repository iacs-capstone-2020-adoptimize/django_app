from django.shortcuts import render
from django.http import HttpResponse
import cv2
from .code.process_video import get_features_frames, score_video_log_2
from .code.video_utils import CatVideo
import numpy as np


def index(request):
    return render(request, 'apa/index.html')


def result(request):
    file = request.FILES['filename']
    key = request.session._get_or_create_session_key()
    cat_video = load_video(file, key)
    opt_img = run_model(cat_video)
    # Note opt_img is RGB, but cv2 expects BGR
    cv2.imwrite(f'apa/media/image_{key}.jpg', opt_img[:, :, ::-1])
    opt_img_path = f'http://127.0.0.1:8000/apa/media/image_{key}.jpg'
    return render(request, 'apa/result.html', {"opt_img": opt_img_path})


# Helper functions
def load_video(file, key):
    dest_file = f'apa/media/video_{key}.mp4'
    with open(dest_file, 'wb+') as dest:
        for chunk in file.chunks():
            dest.write(chunk)
    video = CatVideo(dest_file)
    return video


def run_model(cat_video):
    features = get_features_frames(cat_video.iter_all_frames())
    return cat_video.get_frame_num(score_video_log_2(features))
