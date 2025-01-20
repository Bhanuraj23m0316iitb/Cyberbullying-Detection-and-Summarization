FROM python:3.7
COPY . /app.py , ./definations.py , ./twitter_data
WORKDIR /app
RUN pip install -r requirements.txt
RUN app.py
EXPOSE $PORT

