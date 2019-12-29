import os
from flask import Flask, flash, request, redirect, url_for, jsonify
from werkzeug.utils import secure_filename
import cv2
import numpy as np

from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications import vgg19
from tensorflow.keras import backend as K

from scipy.optimize import fmin_l_bfgs_b
from scipy.misc import imsave

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'])

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/upload', methods = ['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit a empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            #return redirect(url_for('uploaded_file', filename=filename))

    return '''
		    <!doctype html>
		    <title>Upload new File</title>
		    <h1>Upload new File</h1>
		    <form method=post enctype=multipart/form-data>
		      <input type=file name=file>
		      <input type=submit value=Upload>
		    </form>
		    '''


def preprocess_image(image_path, image_height, image_width):
	img = load_img(image_path, target_size=(image_height, image_width))
	img = img_to_array(img)
	img = np.expand_dims(img, axis=0)
	img = vgg19.preprocess_input(img)

	return img


def deprocess_image(x):
	x[:, :, 0] += 103.939
	x[:, :, 1] += 116.779
	x[:, :, 2] += 123.68

	x = x[:, :, ::-1]
	x = np.clip(x, 0, 255).astype('uint8')

	return x


def content_loss(base, combination):
	return K.sum(K.square(combination - base))


def gram_matrix(x):
	features = K.batch_flatten(K.permute_dimensions(x, (2, 0, 1)))
	gram = K.dot(features, K.transpose(features))

	return gram



def style_loss(style, combination, image_height, image_width):
	S = gram_matrix(style)
	C = gram_matrix(combination)
	channels = 3
	size = image_height * image_width

	return K.sum(K.square(S - C)) / (4.0 * (channels ** 2) * size ** 2)


def total_variation_loss(x, image_height, image_width):
	a = K.square(
		x[:, :image_height - 1, :image_width - 1, :] -
		x[:, 1:, :image_width - 1, :])
	b = K.square(
		x[:, :image_height - 1, :image_width - 1, :] -
		x[:, 1:, :image_height - 1, :])

	return K.sum(K.pow(a + b, 1.25))

def nts():

	target_path = 'musk_boii.jpg'
	style_path = 'wave.jpg'

	width, height = load_img(target_path).size
	image_height = 400
	image_width = int(width * image_height / height)
