# Instructions

1. Clone this repo: `git clone https://github.com/iacs-capstone-2020-adoptimize/django_app.git`
2. Install Git Large File Storage. See instructions [here](https://git-lfs.github.com/). Then, inside this repo, run `git lfs pull` to download the object detection model weights.
3. Make sure `python` and `pip` are installed.
4. Ideally in a [virtual environment](https://docs.python.org/3/library/venv.html), install pip version 19, e.g. `pip install pip==19.3.1`.
5. Install the other python requirements using `pip install -r requirements.txt`.
6. Install `ffmpeg`. Use `apt-get` or `brew`or see [here](https://www.ffmpeg.org/download.html) for other options.
7. In `django_app/apa_web`, run the django development server using `python manage.py runserver 0.0.0.0:8000` 
