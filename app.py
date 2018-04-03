# -*- coding: utf-8 -*-
import os
import uuid

from flask import Flask, send_from_directory
from flask import jsonify
from flask import make_response
from flask import render_template
from flask import request

from acmr_models.create_new_info import extract_image_features, extract_text
from acmr_models.db_model import db
from acmr_models.db_model.db_wikipedia import get_wikipedia_with_id
from acmr_models.knn import get_vecs_knn_ret, get_feats_knn_ret
from acmr_models.train_adv_crossmodal_simple_wiki import run_acmr

sqlite_path = os.path.dirname(os.path.realpath(__file__))
print "current_dir : " + sqlite_path
app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:////' + sqlite_path + '/wikipedia.db'
app.config.setdefault('SQLALCHEMY_TRACK_MODIFICATIONS', True)
db.init_app(app)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/img2txt_ret')
def result():
    # get_vecs_knn_ret(vecs_trans[0])
    # get_feats_knn_ret(feats_trans[0])
    search_id = request.args["id"]
    print(search_id)
    # run_vgg16(sess, search_id + ".jpg")
    cat_id, imgs_id = get_feats_knn_ret(run_acmr(1, search_id))
    categories_list = ["art", "biology", "geography", "history", "literature",
                       "media", "music", "royalty", "sport", "warfare"]
    label = categories_list[cat_id - 1]
    imgs = []
    for img_id in imgs_id:
        img = {}
        temp_img = get_wikipedia_with_id(img_id)
        img["pic_id"] = temp_img.pic_id
        img["name"] = temp_img.name
        img["texts"] = temp_img.texts
        imgs.append(img)
    return render_template('img2txt.html', search_id=search_id, label=label, imgs=imgs, categories_list=categories_list)


ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'])


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS


@app.route('/receive', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['input-image']
        if file and allowed_file(file.filename):
            filename = str(uuid.uuid1())  # secure_filename(file.filename)
            file.save(os.path.join('./uploads', filename + ".jpg"))
            extract_image_features(filename)
            return jsonify({'success': filename})
    return jsonify({'fail': 'shibai'})


@app.route('/search_img', methods=["POST"])
def search_img():
    try:
        query_str = request.form["query_str"]
        return make_response("success")
    except Exception:
        return make_response("fail", 400)


@app.route("/txt2img_ret")
def txt2img_ret():
    query_txt = request.args["query"]
    if len(query_txt) > 0:
        cat_id, imgs_id = get_vecs_knn_ret(run_acmr(0, extract_text(query_txt)))
        # import numpy as np
        # cat_id, imgs_id = get_vecs_knn_ret(np.fromstring(get_wikipedia_with_id(20).vecs, np.float32))
        categories_list = ["art", "biology", "geography", "history", "literature",
                           "media", "music", "royalty", "sport", "warfare"]
        label = categories_list[cat_id - 1]
        imgs = []
        for img_id in imgs_id:
            img = {}
            temp_img = get_wikipedia_with_id(img_id)
            img["pic_id"] = temp_img.pic_id
            img["name"] = temp_img.name
            img["texts"] = temp_img.texts
            imgs.append(img)
        return render_template('txt2img.html', query_txt=query_txt, label=label, imgs=imgs,
                               categories_list=categories_list)
    else:
        return make_response("异常输入～")


@app.route("/download/<filename>", methods=['GET'])
def download_file(filename):
    # 需要知道2个参数, 第1个参数是本地目录的path, 第2个参数是文件名(带扩展名)
    directory = "uploads"
    try:
        return send_from_directory(directory, filename, as_attachment=True)
    except Exception:
        return send_from_directory(directory, "new.jpg", as_attachment=True)


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=2333, debug=True)
    # init_all_table()

    # for i in range(2000):
    #     ret = get_img_labels(sess, "../static/multi-label/" + str(i + 1) + ".jpg")
    #     create_new_img_inf(i + 1, ret["desert"], ret["mountains"], ret["sea"], ret["sunset"], ret["trees"])
