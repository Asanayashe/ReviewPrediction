FROM python:3.10

WORKDIR /usr/src/app

COPY requirements.txt ./
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y
RUN pip install --no-cache-dir -r requirements.txt
RUN curl -LOJ https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.5.0/en_core_web_sm-3.5.0.tar.gz
RUN pip install -U en_core_web_sm-3.5.0.tar.gz
RUN rm en_core_web_sm-3.5.0.tar.gz

COPY movie_review ./

CMD ["python", "manage.py", "runserver", "0.0.0.0:8000"]
