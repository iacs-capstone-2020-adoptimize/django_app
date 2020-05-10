# Instructions

1. Clone this repo: `git clone https://github.com/iacs-capstone-2020-adoptimize/django_app.git`.
2. Download the model weights from [here](https://drive.google.com/open?id=1zOqVks99qc3JnIwyI2GpFt2ZA9huVWCs) and move the file to `django_app/apa_web/apa/code/yolo_training/Data/Model_Weights/trained_weights_final.h5`.
3. Make sure `python` and `pip` are installed.
4. Ideally in a [virtual environment](https://docs.python.org/3/library/venv.html), install pip version 19, e.g. `pip install pip==19.3.1`.
5. Install the other python requirements using `pip install -r requirements.txt`.
6. Install `ffmpeg`. Use `apt-get` or `brew`or see [here](https://www.ffmpeg.org/download.html) for other options.
7. In `django_app/apa_web`, run the django development server using `python manage.py runserver 0.0.0.0:8000`.
8. Access the server from `<host>:8000/apa`.
