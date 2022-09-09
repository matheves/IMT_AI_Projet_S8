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

# Setup Jupyter
RUN pip install jupyter -U && pip install jupyterlab

EXPOSE 8888

CMD ["jupyter", "lab", "--ip=0.0.0.0", "--allow-root"]
#CMD ["python", "CartPole_DQN.py"]
#CMD ["sleep", "10000"]