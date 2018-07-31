# Created by wangzixin at 31/07/2018
from mongoengine import *

connect('rs')


class RatingDocument(Document):
    user_key = StringField()
    item_key = StringField()
    value = FloatField()

    meta = {
        'ordering': ['+user_key', '+item_key']
    }


class FeaturesDocument(Document):
    type = StringField()
    key = StringField()
    values = ListField()

    meta = {
        'ordering': ['+type', '+key']
    }


def load_features(key):
    result = FeaturesDocument.objects(key=key).first()
    if result is not None:
        return result.values
    else:
        return None


def save_features(key, values):
    FeaturesDocument.objects(key=key).update_one(values=values, upsert=True)
