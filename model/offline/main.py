# Created by wangzixin at 31/07/2018
import numpy as np
from model.tools.transformer import to_training_data, features2matrix
from model.data.data_pool import DataPool, add_rating, Feature
from model.training.model import Model
from model.data.mongo import *
from model.consts import *


def training():
    # load data
    k_set = set()

    if len(DataPool.pool) > 0:
        DataPool.clear()

    assert len(DataPool.pool) == 0 and len(DataPool.key_pool) == 0

    for r in Ratings.objects:
        k_set.add(r.item_key)
        k_set.add(r.user_key)
        print('adding rating uk = {}, ik = {}, value = {}'.format(r.user_key, r.item_key, r.value))
        add_rating(r.user_key, r.item_key, r.value, load_features)

    assert len(k_set) == len(DataPool.pool)

    # train
    try:
        for epoch in range(OFFLINE_TRAINING_EPOCHS):
            losses = []
            for feature in DataPool.pool:
                sampled = DataPool.sample(feature)
                x, y = to_training_data(sampled)

                assert x.shape[0] == y.shape[0]

                model = Model(theta=feature.values)
                model.fit(x, y, epochs=OFFLINE_SAMPLE_TRAINING_EPOCHS)

                feature.values = model.theta
                losses.append(model.losses[-1])
                # print('loss {} {}'.format(feature.key, model.losses[-1]))

            print('Epoch-{} mean loss = {}'.format(epoch + 1, sum(losses) / len(DataPool.pool)))
    except KeyboardInterrupt:
        print('Interrupted by User ...')

    # store new features
    for feature in DataPool.pool:
        for x in DataPool.pool[feature]:
            p = np.matmul(feature.values, x.feature.values)
            if feature.type == 'user':
                print('Prediction {} * {} = {}'.format(feature.key, x.feature.key, p))
        save_features(feature.key, feature.type, feature.values)


def generate_top_k(k=100):
    items = Features.objects(type='item')[:1000]

    items_matrix, item_keys = features2matrix(items)

    for uf in Features.objects(type='user'):
        user = Feature(uf.key, values=uf.values)

        model = Model(theta=user.values)
        preds = model.predict(items_matrix)

        itempreds = sorted(zip(item_keys, preds), key=lambda x: x[1], reverse=True)

        print('user = ', user.key)
        print('preds = ')
        print(itempreds)

        user_associated_paris = DataPool.get(user.key)
        user_associated_item_key_set = set([pair.feature.key for pair in user_associated_paris])
        limit = k + len(user_associated_paris)
        user_top_limit_item_set = set([item_pred[0] for item_pred in itempreds[:limit]])
        user_top_k_item_set = (user_top_limit_item_set - user_associated_item_key_set)
        # user_top_k_items = [(k, v) for k, v in sorted_item_preds[:limit] if k in user_top_k_item_set]
        user_top_k_items = filter(lambda kv: kv[0] in user_top_k_item_set, itempreds[:limit])

        # update database
        UserTopKItems.objects(user=user.key).update(items=[k for k, v in user_top_k_items], upsert=True)


def main():
    training()
    generate_top_k()


if __name__ == '__main__':
    main()
