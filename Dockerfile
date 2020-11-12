FROM ubuntu:20.04

ENV LANG C.UTF-8
ENV APP_ROOT /app

ENV DEBIAN_FRONTEND noninteractive

RUN mkdir -p $APP_ROOT
WORKDIR $APP_ROOT

RUN ln -sf /usr/share/zoneinfo/Asia/Tokyo /etc/localtime

RUN apt-get update \
    && apt-get upgrade -y \
    && apt-get install -y \
    git \
    make \
    cmake \
    curl \
    xz-utils \
    file \
    sudo \
    build-essential \
    software-properties-common \
    mecab \
    libmecab-dev \
    #   mecab-ipadic-utf8 \
    python3.8 \
    python3.8-dev \
    python3-pip \
    python3.8-venv \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install -U pip
COPY ./requirements.txt .
RUN pip install -r requirements.txt

RUN echo "すもももももももものうち中居正広"|mecab
RUN git clone --depth 1 https://github.com/neologd/mecab-ipadic-neologd.git /tmp/neologd \
    && /tmp/neologd/bin/install-mecab-ipadic-neologd -n -y \
    && rm -rf /tmp/neologd

RUN echo "すもももももももものうち中居正広"|mecab -d /usr/lib/x86_64-linux-gnu/mecab/dic/mecab-ipadic-neologd

RUN mkdir data
COPY data/text.txt data/
COPY *.py ./

CMD ["python3", "app.py"]