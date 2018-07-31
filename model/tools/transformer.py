# Created by wangzixin at 31/07/2018
import numpy as np


def to_training_data(samples: list):
    """

    :param samples: list of Feature
    :return:
        x, y
    """

    assert len(samples) > 0

    x = [item.feature.values for item in samples]
    y = [item.target for item in samples]

    return np.array(x), np.array(y)
