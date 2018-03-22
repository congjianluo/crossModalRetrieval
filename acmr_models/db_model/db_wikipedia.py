# -*- coding: utf-8 -*-
from sqlalchemy import create_engine, Column, Integer, String, text, LargeBinary, Boolean
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base
import numpy as np

engine = create_engine('sqlite:///wikipedia.db', echo=True)
Base = declarative_base()
DBSession = sessionmaker(bind=engine)
session = DBSession()


class WikiPedia(Base):
    __tablename__ = "wikipedia"

    id = Column(Integer, primary_key=True)
    pic_id = Column(String(64))
    document_id = Column(String(64))
    name = Column(String(32))
    texts = Column(String(300))
    feats = Column(LargeBinary(512))
    vecs = Column(LargeBinary(512))
    label = Column(Integer)
    is_test = Column(Boolean)

    def __init__(self, id, pic_id, document_id, name, texts, feats, vecs, label, is_test=0):
        self.id = id
        self.pic_id = pic_id
        self.document_id = document_id
        self.name = name
        self.texts = texts
        self.feats = feats
        self.vecs = vecs
        self.label = label
        self.is_test = is_test


def create_new_img_inf(id, pic_id, document_id, name, texts, feats, vecs, label):
    try:
        new_wiki = WikiPedia(id, pic_id, document_id, name, texts, feats, vecs, label)
        session.add(new_wiki)
        session.commit()
        print(str(id) + "success")
        return True
    except Exception as e:
        print(e)
        return False


def select_wikipedia_info(document_id):
    filter_sql = "document_id=" + "'" + document_id + "'"
    result = session.query(WikiPedia).filter(text(filter_sql)).scalar()
    return result


def select_wikipedia_info_with_pic(pic_id):
    filter_sql = "pic_id=" + "'" + pic_id + "'"
    result = session.query(WikiPedia).filter(text(filter_sql)).scalar()
    return result


def update_wikipedia_info(wikipedia_inf):
    # session.update(wikipedia_inf)
    # print(wikipedia_inf.id)
    session.commit()


def get_all_img_feats():
    results = session.query(WikiPedia).all()
    wikipedia_feats = []
    for item in results:
        wikipedia_feats.append(np.fromstring(item.feats, dtype=np.float32))
    return wikipedia_feats


def get_all_vecs():
    results = session.query(WikiPedia).all()
    wikepedia_vecs = []
    for item in results:
        wikepedia_vecs.append(np.fromstring(item.vecs, dtype=np.float64))
    return wikepedia_vecs


def get_all_label():
    results = session.query(WikiPedia).all()
    wikipedia = []
    for item in results:
        wikipedia.append(item.label)
    return wikipedia


def get_all_wikipedia_dataset():
    results = session.query(WikiPedia).filter().all()
    wikipedia_results = []
    for item in results:
        result = {}
        result["feats"] = np.fromstring(item.feats, dtype=np.float32)
        result["vecs"] = np.fromstring(item.vecs, dtype=np.float64)
        result["label"] = item.label
        wikipedia_results.append(result)
    return wikipedia_results


def get_all_wikipedia():
    results = session.query(WikiPedia).filter().all()
    return results


def select_img_inf(query_list):
    query_len = len(query_list)
    if query_len == 1:
        acc = 0.8
    elif query_len == 2:
        acc = 0.3
    else:
        acc = 0.1

    t = 0
    filter_sql = ""
    for item in query_list:
        t += 1
        if t is not 1:
            filter_sql += " and "
        filter_sql += item + " > " + str(acc)
    print(filter_sql)
    result = session.query(WikiPedia).filter(text(filter_sql)).order_by(text(query_list[0] + " desc")).all()[0:6]
    # imgs = []
    # for item in result:
    #     imgs.append(item.id)
    # print(imgs)
    return result


def init_all_table():
    Base.metadata.create_all(engine)

# if __name__ == "__main__":
#     # new_img = Img(0, "1.2", "1.2", "1.2", "1.2", "1.2")
#     # session.add(new_img)
#     # session.commit()
#     # session.close()
#     # init_all_table()
#     # result = select_wikipedia_info("61e2937e84fcbeb9356af454408616e4-2.11")
#     # a = np.fromstring(result.vecs, dtype=np.float64)
#     # b = np.fromstring(result.feats, dtype=np.float32)
#     # print("aaa")
#     get_all_label()
