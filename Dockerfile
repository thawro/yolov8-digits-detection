# pull official base image
FROM node:18.16.0

# set working directory
WORKDIR /app

RUN userdel -r node

# USER root
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

WORKDIR $HOME/app

COPY --chown=user frontend/ $HOME

RUN npm install

# start app
CMD ["npm", "start"]
