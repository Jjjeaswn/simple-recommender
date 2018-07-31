# Created by wangzixin at 31/07/2018
from model.tools.transformer import to_training_data
from model.data_centre.data_pool import DataPool, add_rating
from model.training.model import Model
import numpy as np
from model.data_centre.mongo import *


def main():
    for r in RatingDocument.objects:
        print('adding rating uk = {}, ik = {}, value = {}'.format(r.user_key, r.item_key, r.value))
        add_rating(r.user_key, r.item_key, r.value, load_features)

    try:
        for _ in range(1000):
            for feature in DataPool.pool:
                sampled = DataPool.sample(feature)
                x, y = to_training_data(sampled)
                model = Model(theta=feature.values)
                model.fit(x, y, epochs=1)
                feature.values = model.theta
                print('loss {} {}'.format(feature.key, model.losses[-1]))
    except KeyboardInterrupt:
        pass

    for feature in DataPool.pool:
        for x in DataPool.pool[feature]:
            p = np.matmul(feature.values, x.feature.values)
            print('Prediction {} * {} = {}'.format(feature.key, x.feature.key, p))
        save_features(feature.key, feature.values)


if __name__ == '__main__':
    main()
