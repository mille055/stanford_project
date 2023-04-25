# Use an official Python runtime as a parent image
FROM python:3.10-slim

RUN apt-get update && \
    apt-get install -y gfortran liblapack-dev libblas-dev

RUN python -m venv /app/venv
ENV PATH="/app/venv/bin:$PATH"

RUN pip install --upgrade pip setuptools wheel
RUN pip install numpy

# Set the working directory to /app
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed packages specified in requirements.txt
RUN pip install --trusted-host pypi.python.org -r requirements.txt

# Make port 8501 available to the world outside this container
EXPOSE 7000

# Define environment variable
ENV NAME World

# Run app.py when the container launches
CMD ["streamlit", "run", "demo/demo.py"]
