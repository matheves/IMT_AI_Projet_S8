FROM xkianteb/mujoco:latest

WORKDIR /app

ENV DOCKER_FLAG=1

ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/root/.mujoco/mujoco210/bin
# Setup MuJoCo
RUN python -m pip install --upgrade pip

RUN pip install pipreqs &&\
    pipreqs . &&\
    pip install psutil &&\
    pip install flask &&\
    pip install pyglet &&\
    pip install mujoco_py &&\
    mkdir -p /root/.mujoco/mujoco210 &&\
    cp -r /mujoco/.mujoco/mujoco210/* /root/.mujoco/mujoco210/ 

RUN apt-key del 7fa2af80
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/3bf863cc.pub
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu2004/x86_64/7fa2af80.pub

RUN apt-get update -y
RUN apt-get install xvfb -y

RUN pip install --user tf-agents[reverb] --no-cache-dir

# Setup Jupyter
RUN pip install jupyter -U && pip install jupyterlab

COPY . .

EXPOSE 5050

CMD ["python", "API/app.py"]

#CMD ["python", "humanoid.py"]

#CMD xvfb-run -a -s "-screen 0 1400x900x24" jupyter notebook --ip=0.0.0.0 --allow-root
