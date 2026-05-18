FROM node:20-alpine

WORKDIR /usr/src/app

# Install dependencies first (cached layer)
COPY api/package*.json ./
RUN npm install

# Copy API source
COPY api/ .

# Copy the dashboard folder so Express can serve it as static files
COPY dashboard/ ../dashboard/

EXPOSE 3000

CMD ["npm", "start"]
