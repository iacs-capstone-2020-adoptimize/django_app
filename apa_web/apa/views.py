from django.shortcuts import render
from django.http import HttpResponse
import cv2
from .code.process_video import get_features_frame_list, score_video_log_2
import numpy as np

def index(request):
    return render(request, 'apa/index.html')

def result(request):
    file = request.FILES['filename']
    frames = load_video(file)
    opt_img = run_model(frames)
    cv2.imwrite('apa/media/opt_img.jpg', opt_img)
    opt_img_path = 'http://127.0.0.1:8000/apa/media/opt_img.jpg'
    return render(request, 'apa/result.html', {"opt_img": opt_img_path})

# Helper functions
def load_video(file):
    dest_file = 'apa/media/video.mp4'
    with open(dest_file, 'wb+') as dest:
        for chunk in file.chunks():
            dest.write(chunk)
    vc = cv2.VideoCapture(dest_file)
    success,image = vc.read()
    frames = []
    while success:
        frames.append(image)
        success,image = vc.read()
    print('# frames =', len(frames))
    return frames

def run_model(frames):
    features = get_features_frame_list(frames)
    return frames[score_video_log_2(features)]
    # feats = get_features_video(frames)
    # print(feats)
    # #frame_idx = score_video_log2(feats)
    # frame_idx = 0
    # return frames[frame_idx]

# def get_features_video(frames, sample_rate=10):
#     cat_frames = list()
#     frame_list = list()
#     for i, frame in enumerate(frames):
#         if i % sample_rate == 0:
#             # features = get_features_frame(frame)
#             features = np.array([0.5,0.5,0.5,1,1,1,1,1,1,20,8])
#             if np.any(features[3:9] != 0):
#                 cat_frames.append(features)
#                 frame_list.append(i)
#     frame_list = np.array(frame_list).reshape((-1, 1))
#     return np.hstack((cat_frames, frame_list))
