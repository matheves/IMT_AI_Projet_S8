# IMT_AI_Projet_S8

## Build & Run Docker image

Open a terminal at the root of your project.

```
docker build -t image_name .
```
```
docker run -p 8888:8888 image_name sleep infinity
```
Then ctrl+clic on the link shown in your terminal

In Docker Desktop, open the container CLI and run this command:

```
xvfb-run -s "-screen 0 1400x900x24" jupyter notebook --ip=0.0.0.0 --allow-root
```

Open the URL for the jupyter notebook in your browser
