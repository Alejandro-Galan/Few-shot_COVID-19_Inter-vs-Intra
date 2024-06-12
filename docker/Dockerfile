# to run docker
# docker run  -it --rm --gpus "device=6" -v /raid/grfia/ajgallego/face_anim/nerf-pytorch:/home/user -u $(id -u):$(id -g) fddbeeee4e8e bash home/user/run_script.sh
FROM tensorflow/tensorflow:latest-gpu

ENV DEBIAN_FRONTEND noninteractive
RUN apt-get -y update \
&& DEBIAN_FRONTEND=noninteractive apt-get -y install sudo curl locales \ 
&& rm -rf /var/lib/apt/lists/*

# set bash as current shell
RUN chsh -s /bin/bash
SHELL ["/bin/bash", "-c"]

# Make sure we use cuda
ENV export FORCE_CUDA="1"

# install useful packages
RUN apt-get update
RUN apt-get install -y git make wget

# User specifics
ARG USER_ID=10008
ARG GROUP_ID=10001

RUN addgroup --gid $GROUP_ID user
RUN adduser --disabled-password --gecos '' --uid $USER_ID --gid $GROUP_ID user
RUN echo "user ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/90-user

ENV HOME=/home/user
RUN chmod 777 /home/user
ENV PATH="/home/user/.local/bin:${PATH}"
WORKDIR /home/user



USER user

# install tensorflow
RUN sudo apt-get update
RUN python3 -m pip install --upgrade pip
RUN pip3 install --upgrade numpy==1.23.5
RUN pip install matplotlib scipy theano
RUN sudo apt install -y python3-opencv libopenexr-dev
RUN pip install scikit-learn scikit-image openexr
RUN pip install tensorflow scikit-learn pandas matplotlib

RUN sudo apt-get update && sudo apt-get install ffmpeg libsm6 libxext6  -y


CMD ["nginx", "-g", "daemon off;"]
