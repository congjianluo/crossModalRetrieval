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
from scipy.misc import imread, imresize

from vgg_module import vgg16

# from download import downloadvvvv
# import nltk
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
    batch_size = 1
    filenames = []
    # for dirname in ["art", "biology", "geography", "history", "literature", "media", "music", "royalty", "sport",
    #                 "warfare"]:
    #     filenames += glob_recursive('../wikipedia_dataset/images/' + dirname, '*.jpg')

    filenames += glob_recursive('./uploads/', pic_name + '.jpg')

    with tf.Session() as sess:
        start = time.time()
        vgg = vgg16.Vgg16(imgs, "./acmr_models/vgg_module/vgg16_weights.npz", sess)
        print('Loaded vgg16 in %4.4fs' % (time.time() - start))

        img_data = [imresize(imread(filename, mode='RGB'), (224, 224)) for filename in filenames]
        feats = sess.run(vgg.fc2, feed_dict={vgg.imgs: img_data})
        # wikipedia_info = select_wikipedia_info_with_pic(splitext(basename(batch_filenames[ii]))[0])
        img_feats = feats[0]
        # wikipedia_info.feats = feats[ii].tostring()  # np.fromstring(my_string, dtype=np.float)
        # update_wikipedia_info(wikipedia_info)
        print('pic_feats finished in %4.4fs' % (time.time() - start))

        # batch_filenames = filenames[num_batches * batch_size: len(filenames)]
        # img_data = [imresize(imread(filename, mode='RGB'), (224, 224)) for filename in batch_filenames]
        # feats = sess.run(vgg.fc2, feed_dict={vgg.imgs: img_data})
        # for ii in range(len(batch_filenames)):
        #     img_feats[splitext(basename(batch_filenames[ii]))[0]] = feats[ii]
        #     wikipedia_info = select_wikipedia_info(splitext(basename(batch_filenames[ii]))[0])
        #     wikipedia_info.feats = feats[ii].tostring()  # np.fromstring(a)
        #     update_wikipedia_info(wikipedia_info)
        # print('[%d/%d] - finished in %4.4fs' % (len(filenames), len(filenames), time.time() - start))
        # img_feats_list = img_feats.values()
    cPickle.dump(img_feats, open('./acmr_models/feats/' + pic_name + '-feats.pkl', 'wb'), cPickle.HIGHEST_PROTOCOL)
    # cPickle.dump(img_feats_list, open('./data/wikipedia_dataset/img_feats_list_vgg16.pkl', 'wb'), cPickle.HIGHEST_PROTOCOL)
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
    # # init_all_table()
    # print('Extracting wiki text')
    #
    # all_xml_files = glob(os.path.join('../wikipedia_dataset/texts', '*.xml'))
    # text_words_map = {}
    # all_words = []
    #
    # # Extract all words and unique words
    # start = time.time()
    # i = 0
    # for xml_filename in all_xml_files:
    #     basename_without_ext = splitext(basename(xml_filename))[0]
    #     tokens, texts, name, pic_id = extract_words_from_xml(xml_filename)
    #     # create_new_img_inf(i, pic_id, basename_without_ext, name, texts, "", "", -1)
    #     i += 1
    #
    #     text_words_map[basename_without_ext] = tokens
    #     all_words += tokens
    # unique_words = list(set(all_words))
    # # cPickle.dump(unique_words, open('./data/wikipedia_dataset/unique_words.pkl', 'wb'), cPickle.HIGHEST_PROTOCOL)
    # print('Extracted %d unique words in %4.4fs' % (len(unique_words), time.time() - start))
    # # cPickle.dump(text_words_map, open('./data/wikipedia_dataset/text_words_map.pkl', 'wb'), cPickle.HIGHEST_PROTOCOL)
    #
    # # Build dictionary
    # cPickle.dump(text_words_map, open('./images/text_words_map.pkl', 'wb'), cPickle.HIGHEST_PROTOCOL)
    text_words_map = cPickle.load(open('./images/all_text_words_map.pkl', 'rb'))

    dictionary = gensim.corpora.Dictionary(tokens for _, tokens in text_words_map.items())
    dictionary.filter_extremes(no_below=5, keep_n=5000)
    dictionary.compactify()
    tfidf = gensim.models.tfidfmodel.TfidfModel([tokens for _, tokens in text_words_map.items()], dictionary=dictionary)
    print('%s' % len(dictionary.keys()))
    print('%s' % len(text_words_map.keys()))

    # Generate word vec representations
    chunksize = 5000
    print('Calculating vector representations')
    start = time.time()
    text_vecs_map = {}
    i = 0
    input_text = text_words_map["0b4ebd99673d910a6747df881d000dc1-9"]
    tokens = input_text
    bow = np.zeros([len(dictionary.keys())])
    # cnt_num = 0g
    for chunk in gensim.utils.chunkize(tokens, chunksize, maxsize=2):
        vec_list = dictionary.doc2bow(chunk)
        vec_list_tfidf = tfidf[vec_list]
        keys = [key for key, _ in vec_list_tfidf]
        vec_dict = {key: val for key, val in vec_list}
        for key in keys:
            bow[key] += vec_dict[key]
        # wikipedia_info = select_wikipedia_info(k)
        # wikipedia_info.vecs = bow.tostring()
        # update_wikipedia_info(wikipedia_info)
        # print(i)
        # i += 1
    cPickle.dump(bow, open('./images/bow.pkl', 'wb'), cPickle.HIGHEST_PROTOCOL)
    # cPickle.dump(text_vecs_map, open('./data/wikipedia_dataset/filename_vecs_map.pkl', 'wb'), cPickle.HIGHEST_PROTOCOL)
    print('Calculated vector representations in %4.4fs' % (time.time() - start))


def main():
    # download_vgg16_weights('./data/vgg16')
    extract_text("")
    # extract_image_features("0b357ab251ab2ee706a23b6d8fcdd726")
    # create_train_test_sets()


if __name__ == '__main__':
    main()
