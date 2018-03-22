# -*- coding: utf-8 -*-
import tensorflow as tf
from models.adv_crossmodal_simple_wiki import AdvCrossModalSimple, ModelParams


# from models.wiki_shallow import AdvCrossModalSimple, ModelParams
def run_acmr(feats=0, query_str=None):
    if query_str is None:
        return False

    graph = tf.Graph()
    model_params = ModelParams()
    model_params.update()

    with graph.as_default():
        model = AdvCrossModalSimple(model_params)
        saver = tf.train.Saver()

    with tf.Session(graph=graph) as sess:
        # model.train(sess)
        # saver.save(sess, "./simple/model2.ckpt")
        saver.restore(sess, "./acmr_models/simple/model2.ckpt")
        if feats == 1:
            return model.eval_feats(sess, query_str)
        else:
            return model.eval_vecs(sess, query_str)
        # model.eval_random_rank()


if __name__ == '__main__':
    run_acmr(1, "")
