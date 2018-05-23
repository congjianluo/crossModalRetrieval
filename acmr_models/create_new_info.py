# -*- coding: utf-8 -*-
from __future__ import print_function

import cPickle
import fnmatch
import json
import os
import time
from glob import glob
from os.path import basename, splitext

import gensim.utils
import numpy as np
import tensorflow as tf
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.tag import _pos_tag, PerceptronTagger
from scipy.misc.pilutil import imread, imresize

from vgg_module import vgg16

# from download import downloadvvvv
# import nltk
# nltk.download('averaged_perceptron_tagger')
# nltk.download('stopwords')

_perceptronTagger = PerceptronTagger()


def tokenize(raw_content, part_tag='NN'):
    tokens = list(gensim.utils.tokenize(raw_content, lowercase=True, deacc=True,
                                        errors='strict', to_lower=True, lower=True))
    standard_stopwords = stopwords.words('english')
    tokens = [word for word in tokens if word.lower() not in standard_stopwords]

    if part_tag is not None:
        tokens = [ww for ww, p in _pos_tag(tokens, None, _perceptronTagger) if p == part_tag]
    return tokens


def glob_recursive(pathname, pattern):
    files = []
    for root, dirnames, filenames in os.walk(pathname):
        for filename in fnmatch.filter(filenames, pattern):
            files.append(os.path.join(root, filename))
    return files


def find_word_vecs(words, data_file):
    vecs = {}
    words = set([w.lower() for w in words])
    with open(data_file, 'r') as f:
        for line in f:
            tokens = line.split()
            if tokens[0] in words:
                vecs[tokens[0]] = np.array([float(num) for num in tokens[1:]], np.float32)
    return vecs


# def download_vgg16_weights(dirpath):
#     url = 'http://www.cs.toronto.edu/~frossard/vgg16/vgg16_weights.npz'
#     if not os.path.exists(os.path.join(dirpath, 'vgg16_weights.npz')):
#         os.mkdir(dirpath)
#         download(url, dirpath)
#     else:
#         print('Found vgg16 weight, skip')


def extract_image_features(pic_name):
    imgs = tf.placeholder(tf.float32, [None, 224, 224, 3])
    filenames = []

    filenames += glob_recursive('./uploads/', pic_name + '.jpg')

    with tf.Session() as sess:
        start = time.time()
        vgg = vgg16.Vgg16(imgs, "./acmr_models/vgg_module/vgg16_weights.npz", sess)
        print('Loaded vgg16 in %4.4fs' % (time.time() - start))

        img_data = [imresize(imread(filename, mode='RGB'), (224, 224)) for filename in filenames]
        feats = sess.run(vgg.fc2, feed_dict={vgg.imgs: img_data})
        img_feats = feats[0]
        print('pic_feats finished in %4.4fs' % (time.time() - start))

    cPickle.dump(img_feats, open('./acmr_models/feats/' + pic_name + '-feats.pkl', 'wb'), cPickle.HIGHEST_PROTOCOL)
    print('Finished')


def extract_words_from_xml(filename):
    with open(filename, 'r') as f:
        soup = BeautifulSoup(f, 'html.parser')
        text = ''
        for t in soup.find_all('text'):
            text += t.string

        name = soup.document.get("name")

        pic_id = soup.find("imageset").image.get("id")

        # convert to unicode and remove additional line breaks
        text = gensim.utils.to_unicode(text)
        text = gensim.utils.decode_htmlentities(text)
        text = text.replace('\n', ' ')
        text_backup = text

        # remove stop words and punctuations
        # tokens = list(gensim.utils.tokenize(text, lowercase=True, deacc=True,
        #                                    errors='strict', to_lower=True, lower=True))
        # standard_stopwords = stopwords.words('english')
        tokens = tokenize(text)
        tokens = [word.encode('utf-8') for word in tokens]
    return tokens, text_backup, name, pic_id


def extract_text(input_text):
    text_words_map = cPickle.load(open('./acmr_models/images/all_text_words_map.pkl', 'rb'))

    dictionary = gensim.corpora.Dictionary(tokens for _, tokens in text_words_map.items())
    dictionary.filter_extremes(no_below=5, keep_n=5000)
    dictionary.compactify()
    tfidf = gensim.models.tfidfmodel.TfidfModel([tokens for _, tokens in text_words_map.items()], dictionary=dictionary)

    chunksize = 5000
    start = time.time()
    text = gensim.utils.to_unicode(input_text)
    text = gensim.utils.decode_htmlentities(text)
    text = text.replace('\n', ' ')
    tokens = tokenize(text)
    tokens = [word.encode('utf-8') for word in tokens]
    bow = np.zeros([len(dictionary.keys())])
    for chunk in gensim.utils.chunkize(tokens, chunksize, maxsize=2):
        vec_list = dictionary.doc2bow(chunk)
        vec_list_tfidf = tfidf[vec_list]
        keys = [key for key, _ in vec_list_tfidf]
        vec_dict = {key: val for key, val in vec_list}
        for key in keys:
            bow[key] += vec_dict[key]
    print('Calculated vector representations in %4.4fs' % (time.time() - start))
    return bow


def main():
    # download_vgg16_weights('./data/vgg16')
    extract_text("")
    # extract_image_features("0b357ab251ab2ee706a23b6d8fcdd726")
    # create_train_test_sets()


if __name__ == '__main__':
    main()
