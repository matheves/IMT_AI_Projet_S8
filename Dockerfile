FROM xkianteb/mujoco:latest

WORKDIR /app

ENV DOCKER_FLAG=1

COPY . .

RUN export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/root/.mujoco/mujoco210/bin

RUN pip install pipreqs &&\
    pipreqs . &&\
    pip install pyglet &&\
    pip install mujoco_py &&\
    mkdir -p /root/.mujoco/mujoco210 &&\
    cp -r /mujoco/.mujoco/mujoco210/* /root/.mujoco/mujoco210/
    
CMD ["python", "CartPole_DQN.py"]
#CMD ["sleep", "10000"]