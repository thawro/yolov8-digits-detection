FROM python:3.11
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y
WORKDIR /app

RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

WORKDIR $HOME/app

COPY --chown=user . $HOME/app


RUN pip3.11 install --no-cache-dir --upgrade -r $HOME/app/requirements.txt


CMD ["gradio", "src/gradio_app.py"]