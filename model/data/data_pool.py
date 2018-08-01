# Created by wangzixin at 31/07/2018

from collections import defaultdict
import random
import numpy as np
from model.consts import LATENT_DIM, INITIAL_THETA_SCALE


class DataPool(object):
    """

    """
    pool = defaultdict(list)
    key_pool = dict()

    @staticmethod
    def _store_obj_key(key):
        """

        Args:
            key (str | Feature):

        Returns:

        Raises:
            ValueError
        """

        if isinstance(key, str) and key not in DataPool.key_pool:
            raise ValueError('key {}\'s obj can not be found in key pool, call `DataPool.add_key(obj)` before this')
        else:
            str_key = str(key)
            if str_key not in DataPool.key_pool:
                DataPool.key_pool[str_key] = key

    @staticmethod
    def add_key(key):
        """

        Args:
            key (Feature):

        Returns:

        """
        DataPool._store_obj_key(key)

    @staticmethod
    def add(key, values=None):
        """

        Args:
            key (str | Feature):
            values(list of Pair):

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
            key (str | Feature):
            values(list of Pair):

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
            key (str | Feature):
            value (Pair):

        Returns:

        """

        DataPool._store_obj_key(key)

        DataPool.pool[key].append(value)

    @staticmethod
    def sample(key, size=32):
        """

        Args:
            key (str | Feature):
            size (int):

        Returns:
            list of Feature:
        """

        if key is str and key in DataPool.key_pool:
            key = DataPool.key_pool[key]

        values = DataPool.pool[key]

        if len(values) > size:
            return random.sample(values, size)

        random.shuffle(values)
        return values

    @staticmethod
    def get(key):
        """

        Args:
            key (str | Feature):

        Returns:
            list of Pair: Paris associated with the key

        """

        if isinstance(key, str):
            key = DataPool.key_pool[key]

        return DataPool.pool[key]

    @staticmethod
    def clear():

        DataPool.pool.clear()
        DataPool.key_pool.clear()


class Feature(object):
    """

    self.values is  np.ndarray
    """

    def __init__(self, key: str, ftype: str = None, values: np.ndarray = None):
        """Constructor for Feature

        """
        self.key = key
        self.type = ftype
        if values is None:
            values = np.random.random((LATENT_DIM,)) * INITIAL_THETA_SCALE
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

    def __str__(self) -> str:
        return str(self.feature)


def add_rating(uk, ik, rating, load_feature_func=None):
    """

    Args:
        uk (str):
        ik (str):
        rating (float):
        load_feature_func (func):

    Returns:
        None
    """

    if uk not in DataPool.key_pool:
        user = Feature(uk, 'user', load_feature_func(uk))
    else:
        user = DataPool.key_pool[uk]

    if ik not in DataPool.key_pool:
        item = Feature(ik, 'item', load_feature_func(ik))
    else:
        item = DataPool.key_pool[ik]

    DataPool.append(item, Pair(user, rating))
    DataPool.append(user, Pair(item, rating))
