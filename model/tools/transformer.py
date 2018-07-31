# Created by wangzixin at 31/07/2018
import numpy as np


def to_training_data(samples: list):
    """

    :param samples: list of Pair
    :return:
        x, y
    """

    assert len(samples) > 0

    x = [item.feature.values for item in samples]
    y = [item.target for item in samples]

    return np.array(x), np.array(y)


def features2matrix(feature_list):
    """

    Args:
        :param feature_list: list of Feature

    Returns:

    """
    matrix = np.array([feature.values for feature in feature_list], dtype=float)
    key_lst = [feature.key for feature in feature_list]

    return matrix, key_lst
