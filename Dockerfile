FROM xkianteb/mujoco:latest

WORKDIR /app

ENV DOCKER_FLAG=1

COPY . .

ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/root/.mujoco/mujoco210/bin

# Setup MuJoCo
RUN pip install pipreqs &&\
    pipreqs . &&\
    pip install pyglet &&\
    pip install mujoco_py &&\
    mkdir -p /root/.mujoco/mujoco210 &&\
    cp -r /mujoco/.mujoco/mujoco210/* /root/.mujoco/mujoco210/ 

RUN apt-key del 7fa2af80
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/3bf863cc.pub
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu2004/x86_64/7fa2af80.pub

RUN apt-get update -y
RUN apt-get install xvfb -y
# Setup Jupyter
RUN pip install jupyter -U && pip install jupyterlab

EXPOSE 8888
