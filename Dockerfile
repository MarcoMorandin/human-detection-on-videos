FROM python:latest

COPY ./requirements.txt /app/requirements.txt

RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

WORKDIR /app

RUN pip install -r requirements.txt

COPY . /app

ENV FLASK_APP server

EXPOSE 8080

CMD [ "python3", "-m" , "flask", "run",  "--host=0.0.0.0", "--port=8080" ]