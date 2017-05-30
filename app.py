import os

from flask import Flask, render_template, request, redirect, url_for, send_from_directory, Response
from werkzeug.utils import secure_filename

import cv2
# import numpy as np

import subprocess

# Initialize the Flask app
app = Flask(__name__)

# Set the upload directory
app.config['UPLOAD_FOLDER'] = 'uploads/'

# Set allowed extensions
app.config['ALLOWED_EXTENSIONS'] = set(['bmp', 'dib', 'jpeg', 'jpg', 'jpe', 'jp2', 'png',
                                        'webp', 'pbm', 'pgm', 'ppm', 'sr', 'ras', 'tiff',
                                        'tif'])


# Check the extension function
def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1] in app.config['ALLOWED_EXTENSIONS']


# Get information about video
def makeNumOdd(num):
    if num % 2 != 0:
        return num
    else:
        return num + 1


def getKSizes(width, height):
    kSizeX = makeNumOdd(int((width / 100)))
    kSizeY = makeNumOdd(int((height / 100)))

    return kSizeX, kSizeY


# Homepage
@app.route('/')
def index():
    return render_template('index.html')


# Picture uploads page
@app.route('/uploads')
def uploads():
    return render_template('uploads.html')


# Neural network uploads page
@app.route('/neural')
def neural():
    return render_template('neural.html')


# Process the file upload
@app.route('/upload', methods=['POST'])
def upload():
    # Get name of the uploaded file
    file = request.files['file']

    # Check the extension
    if file and allowed_file(file.filename):
        # Make filename secured
        filename = secure_filename(file.filename)

        # Save the file
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

        # Process file with OpenCV
        original = cv2.imread(app.config['UPLOAD_FOLDER'] + filename)

        originalHeight, originalWidth, channels = original.shape
        kSizeX, kSizeY = getKSizes(originalWidth, originalHeight)
        blurGaussian = cv2.GaussianBlur(original, (kSizeX, kSizeY), 0)

        d = 1
        blurGaussianBilateral = cv2.bilateralFilter(blurGaussian, d, d * 2, d * 2)

        cv2.imwrite(app.config['UPLOAD_FOLDER'] + filename, blurGaussianBilateral)

        # Redirect user to uploaded file's page
        return redirect(url_for('uploaded_file', filename=filename))

# Process the file upload (neural network)
@app.route('/neural_network', methods=['POST'])
def neural_network():
    # Get name of the uploaded file
    file = request.files['file']

    # Check the extension
    if file and allowed_file(file.filename):
        # Make filename secured
        filename = secure_filename(file.filename)

        # Save the file
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

        # Process file with neural network
        subprocess.call("python3 ./neural-enhance/enhance.py --type=photo --model=repair --zoom=1 --device=gpu0 ./"+ app.config['UPLOAD_FOLDER'] + filename, shell=True)
        #print("python3 ./neural-enhance/enhance.py --type=photo --model=repair --zoom=1 --device=gpu0 ./"+ app.config['UPLOAD_FOLDER'] + filename)

        # Redirect user to uploaded file's page
        return redirect(url_for('uploaded_file', filename=filename.rsplit('.')[0] + "_ne1x.png"))


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


### ### ### ###
@app.route('/video')
def video():
    # Video streaming home page
    return render_template('videostream.html')


def gen():
    # Video streaming generator function
    # Open the video capture
    #vc = cv2.VideoCapture(0)
    vc = cv2.VideoCapture('/home/max/coursework/video/FC6.mp4')

    # Check the capture
    if (not vc.isOpened()):
        print('Video Capture opening Error')

    # Get total number of frames (for loop)
    nFrames = int(vc.get(cv2.CAP_PROP_FRAME_COUNT))

    # Get fps for waitKey function
    fps = vc.get(cv2.CAP_PROP_FPS)
    waitPerFrameInMillisec = int(1000 / fps)

    # Get width and height
    capWidth = vc.get(cv2.CAP_PROP_FRAME_WIDTH)
    capHeight = vc.get(cv2.CAP_PROP_FRAME_HEIGHT)

    for f in range(nFrames):
        # Get the frame
        ret, frame = vc.read()

        # Process it
        # blur = cv2.GaussianBlur(frame, (1,1), 0)
        # d = 1
        # blur = cv2.bilateralFilter(frame, d, d * 2, d * 2)

        kSizeX = makeNumOdd(int(capWidth / 100))
        kSizeY = makeNumOdd(int(capHeight / 100))

        blur = cv2.GaussianBlur(frame, (kSizeX, kSizeY), 0)

        # Show the frame
        cv2.imwrite('t.jpg', blur)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + open('t.jpg', 'rb').read() + b'\r\n')

        # Wait for the processing of next frame OR for pressing 'q' key
        if (cv2.waitKey(waitPerFrameInMillisec) & 0xFF) == ord('q'):
            break


@app.route('/video_feed')
def video_feed():
    """Video streaming route. Put this in the src attribute of an img tag."""
    return Response(gen(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')
### ### ### ###

if __name__ == '__main__':
    app.run()