from flask import Flask, render_template, request, redirect, url_for

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/process_images', methods=['POST'])
def process_images():
    folder = request.files['folder']
    # Process images using your machine learning model
    # Save processed images and metadata
    return redirect(url_for('processed_images'))

@app.route('/processed_images')
def processed_images():
    return render_template('processed_images.html')

@app.route('/filtered_images', methods=['POST'])
def filtered_images():
    image_type = request.form['image_type']
    # Filter images based on the selected image type
    # Render
