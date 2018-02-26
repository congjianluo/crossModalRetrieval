# -*- coding: utf-8 -*-
from sqlalchemy import create_engine, Column, Integer, String
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


def init_all_table():
    Base.metadata.create_all(engine)


# if __name__ == "__main__":
#     new_img = Img(0, "1.2", "1.2", "1.2", "1.2", "1.2")
#     session.add(new_img)
#     session.commit()
#     session.close()
