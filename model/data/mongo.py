# Created by wangzixin at 31/07/2018
from mongoengine import *

connect('rs')


class Ratings(Document):
    user_key = StringField()
    item_key = StringField()
    value = FloatField()

    meta = {
        'ordering': ['+user_key', '+item_key']
    }


class Features(Document):
    type = StringField()
    key = StringField()
    values = ListField()

    meta = {
        'ordering': ['+type', '+key']
    }


class Predictions(Document):
    user_key = StringField()
    item_key = StringField()
    value = FloatField()


class UserTopKItems(Document):
    user = StringField()
    items = ListField()


def load_features(key):
    result = Features.objects(key=key).first()
    if result is not None:
        return result.values
    else:
        return None


def save_features(key, ftype, values):
    Features.objects(key=key, type=ftype).update_one(values=values, upsert=True)
