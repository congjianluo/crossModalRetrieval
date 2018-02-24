# -*- coding: utf-8 -*-
import os

from data.vgg16 import run_vgg16
from inception.label_image import get_img_labels

from flask import Flask
from flask import render_template
from flask import request
from flask import jsonify
from flask import make_response
from werkzeug.utils import secure_filename

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/result')
def result():
    run_vgg16(),
    return render_template('result.html')


ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'])


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS


@app.route('/receive', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['input-image']
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join('./uploads', 'new.jpg'))
            return jsonify({'success': 'chenggong'})
    return jsonify({'fail': 'shibai'})


@app.route('/search_img', methods=["POST"])
def search_img():
    try:
        query_str = request.form["query_str"]
        return make_response("success")
    except Exception:
        return make_response("fail", 400)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
