FROM python:3.8-slim-buster
LABEL maintainer="Tien Thien <tienthienhd@gmail.com>"

# install build utilities
RUN apt-get update \
    && apt-get install -y build-essential cmake libgl1-mesa-glx libgtk2.0-dev

RUN apt-get install -y libcaffe-cpu-dev libopencv-dev \
    libboost-all-dev \
    libopenblas-dev \
    libgflags-dev libgoogle-glog-dev liblmdb-dev \
    libprotobuf-dev protobuf-compiler

COPY requirements.txt .

# install dependencies
RUN pip install --no-cache -r requirements.txt

COPY . /workspace

WORKDIR /workspace/modules/face/PCN_detector

RUN make && make install

# Remove build-essential
RUN apt-get autoremove build-essential cmake -y

# Set the working directory for container
WORKDIR /workspace

#CMD ["python3", "run.py"]
CMD ["uwsgi", "face_lvt.ini"]
#CMD ["python3", "face_detect_test.py"]