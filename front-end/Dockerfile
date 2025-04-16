# Use Node.js as the base image
FROM node:18-alpine

# Set working directory
WORKDIR /app

# Copy package files
COPY package.json .
COPY package-lock.json .

# Install dependencies
RUN npm install

# Copy the rest of the application
COPY . .

# Expose the port the app will run on
EXPOSE 3000

# Start the development server with hot-reloading
CMD ["npm", "start"]