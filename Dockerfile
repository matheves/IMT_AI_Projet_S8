FROM tensorflow/tensorflow:latest-gpu
WORKDIR /app
ENV DOCKER_FLAG=1
COPY . .

RUN apt update &&\
	apt install vim -y &&\
	apt install --upgrade pip &&\
	apt install libosmesa6-dev libgl1-mesa-glx libglfw3 -y &&\
	apt install patchelf &&\
	pip install imageio &&\
	pip install ipython &&\
	pip install matplotlib &&\
	pip install --user tf-agents[reverb] &&\
	pip3 install -U 'mujoco-py<2.2,>=2.1' &&\
	apt install wget &&\
	wget https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz &&\
	tar -xvf mujoco210-linux-x86_64.tar.gz &&\
	mkdir -p ~/.mujoco &&\
	mv mujoco210 ~/.mujoco/ &&\
	echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/root/.mujoco/mujoco210/bin' >> ~/.bashrc &&\
	export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/root/.mujoco/mujoco210/bin

