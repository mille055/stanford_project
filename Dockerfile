# Use an official Python runtime as a parent image
FROM python:3.10


# Set the working directory to /app
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY requirements.txt ./

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Make port 8501 available to the world outside this container
EXPOSE 7000

# Define environment variable
ENV NAME World

# Run app.py when the container launches
CMD ["streamlit", "run", "demo/demo.py"]
