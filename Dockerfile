# Use an official Python runtime as a parent image
FROM python:3.10-slim


RUN pip install --upgrade pip setuptools wheel
RUN pip install numpy

# Set the working directory to /app
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

RUN pip install -r requirements.txt

# Make port 8501 available to the world outside this container
EXPOSE 7000

# Define environment variable
ENV NAME World

# Run app.py when the container launches
CMD ["streamlit", "run", "demo/demo.py"]
