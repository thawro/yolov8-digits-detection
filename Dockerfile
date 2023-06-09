# pull official base image
FROM node:18.16.0

# set working directory
WORKDIR /app

RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

WORKDIR $HOME/app


# add `/app/node_modules/.bin` to $PATH
ENV PATH /app/node_modules/.bin:$PATH

# install app dependencies
COPY frontend/ $HOME

RUN npm install

# start app
CMD ["npm", "start"]
