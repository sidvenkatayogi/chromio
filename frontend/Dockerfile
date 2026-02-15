FROM node:18-alpine

WORKDIR /app

# Install dependencies
COPY package*.json ./
RUN npm install

# Copy source and build
COPY . .
# This creates a /dist folder with static files
RUN npm run build 

# Install a simple static file server
RUN npm install -g serve

EXPOSE 5173

# Serve the 'dist' folder on port 5173
CMD ["serve", "-s", "dist", "-l", "5173"]