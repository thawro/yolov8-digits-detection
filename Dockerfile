# pull official base image
FROM node:18.16.0

# set working directory
WORKDIR /app

RUN userdel -r node

USER root

COPY frontend/ ./


RUN npm install

# start app
CMD ["npm", "start"]
