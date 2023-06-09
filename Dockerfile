# pull official base image
FROM node:18.16.0-alpine

# set working directory
WORKDIR /app


# add `/app/node_modules/.bin` to $PATH
ENV PATH /app/node_modules/.bin:$PATH

# install app dependencies
COPY frontend/ ./

RUN npm install

# start app
CMD ["npm", "start"]
