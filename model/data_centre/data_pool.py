# Created by wangzixin at 31/07/2018

from collections import defaultdict
import random
import numpy as np
from model.consts import LATENT_DIM


class DataPool(object):
    """

    """
    pool = defaultdict(list)
    key_pool = dict()

    @staticmethod
    def _store_obj_key(key):
        """ mapping str(key) to key obj

        Args:


        Returns:

        """
        if key is not str:
            str_key = str(key)
            if str_key not in DataPool.key_pool:
                DataPool.key_pool[str_key] = key

    @staticmethod
    def add(key, values=None):
        """

        Args:
            key, values

        Returns:

        """

        DataPool._store_obj_key(key)

        # key to values
        if values is not None:
            DataPool.pool[key] = values

    @staticmethod
    def join(key, values=None):
        """

        Args:


        Returns:

        """
        DataPool._store_obj_key(key)

        # key to values
        if values is not None:
            DataPool.pool[key] += values

    @staticmethod
    def append(key, value):
        """

        Args:
            key, value

        Returns:

        """

        DataPool._store_obj_key(key)

        DataPool.pool[key].append(value)

    @staticmethod
    def sample(key, size=32):
        """

        Args:
            key, size

        Returns:
           :rtype: list of Bundle
        """
        if key is str and key in DataPool.key_pool:
            key = DataPool.key_pool[key]

        values = DataPool.pool[key]

        if len(values) > size:
            return random.sample(values, size)

        random.shuffle(values)
        return values


class Feature(object):
    """

    self.values is  np.ndarray
    """

    def __init__(self, key: str, values: np.ndarray = None):
        """Constructor for Feature

        """
        self.key = key
        if values is None:
            values = np.random.random((LATENT_DIM,)) * 0.2
        else:
            values = np.array(values, dtype=float)
        self.values = values

    def __str__(self):
        return self.key


class Pair(object):
    """"""

    def __init__(self, feature: Feature, target: float):
        """Constructor for Pair"""

        self.feature = feature
        self.target = target


def add_rating(uk, ik, rating, load_feature_func=None):
    """

    :param uk: user key
    :param ik: item key
    :param float rating : rating
    :param load_feature_func:
    :return:
    """
    user = Feature(uk, load_feature_func(uk))
    item = Feature(ik, load_feature_func(ik))
    DataPool.append(item, Pair(user, rating))
    DataPool.append(user, Pair(item, rating))