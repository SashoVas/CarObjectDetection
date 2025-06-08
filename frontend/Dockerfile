### STAGE 1: Build ###
# Use a Node.js image to build the Angular application
FROM node:20-alpine AS build

# Set the working directory
WORKDIR /app

# Copy package.json and package-lock.json
COPY package*.json ./

# Install dependencies
RUN npm install

# Copy the rest of the application files
COPY . .

# Build the application
RUN npm run build -- --configuration production

### STAGE 2: Serve ###
# Use a lightweight Nginx image to serve the built files
FROM nginx:stable-alpine

# Copy the build output from the build stage
COPY --from=build /app/dist/car-object-detection-frontend/browser /usr/share/nginx/html

# Copy the Nginx configuration file
COPY nginx.conf /etc/nginx/conf.d/default.conf

# Expose port 80
EXPOSE 80 