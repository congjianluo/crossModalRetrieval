# -*- coding: utf-8 -*-
from sqlalchemy import create_engine, Column, Integer, String, text
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base

engine = create_engine('sqlite:///img.db', echo=True)
Base = declarative_base()
DBSession = sessionmaker(bind=engine)
session = DBSession()


class Img(Base):
    __tablename__ = "img"

    id = Column(Integer, primary_key=True)
    desert = Column(String(8))
    mountains = Column(String(8))
    sea = Column(String(8))
    sunset = Column(String(8))
    trees = Column(String(8))

    def __init__(self, id, desert, mountains, sea, sunset, trees):
        self.id = id
        self.desert = desert
        self.mountains = mountains
        self.sea = sea
        self.sunset = sunset
        self.trees = trees


def create_new_img_inf(id, desert, mountains, sea, sunset, trees):
    try:
        new_img = Img(id, desert, mountains, sea, sunset, trees)
        session.add(new_img)
        session.commit()
        print(str(id) + "success")
        return True
    except Exception:
        return False


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
    result = session.query(Img).filter(text(filter_sql)).order_by(text(query_list[0] + " desc")).all()[0:6]
    # imgs = []
    # for item in result:
    #     imgs.append(item.id)
    # print(imgs)
    return result


def init_all_table():
    Base.metadata.create_all(engine)


if __name__ == "__main__":
    # new_img = Img(0, "1.2", "1.2", "1.2", "1.2", "1.2")
    # session.add(new_img)
    # session.commit()
    # session.close()
    select_img_inf(["sea", "desert"])
