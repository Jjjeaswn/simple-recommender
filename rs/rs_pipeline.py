# Created by wangzixin at 02/08/2018

import numpy as np
import multiprocessing
from rs.data.mongo import *
from rs.data.pool import Pool, Drop
from rs.model.cf import CFModel
from rs.framework.pipline import Pipeline

CPU_COUNT = multiprocessing.cpu_count()

# model super parameters
LATENT_DIM = 10
LEARNING_RATE = .1
WEIGHTS_REGULAR = .001
INITIAL_THETA_SCALE = .2

# training
OFFLINE_TRAINING_EPOCHS = 50
OFFLINE_SAMPLE_TRAINING_EPOCHS = 2


def random_weights(latent_dim, scale=.2):
    return np.random.random(latent_dim) * scale


def features2matrix(feature_list):
    """

    Args:
        feature_list (list of Feature):

    Returns:
        (np.ndarray,  list of str): matrix and list of key of features
    """
    matrix = np.array([feature.values for feature in feature_list], dtype=float)
    key_lst = [feature.key for feature in feature_list]
    return matrix, key_lst


class RsPipeline(Pipeline):
    """"""

    def __init__(self, ):
        """Constructor for RsPipeline"""
        super(RsPipeline, self).__init__()
        self.pool = Pool()

    def on_loading(self):
        ratings = Ratings.objects
        cnt = 0
        for rating in ratings:
            if cnt > 1000:
                break
            self.log_info('Loading ... ', cnt)
            cnt += 1
            user = rating.user_key
            item = rating.item_key
            r = rating.value

            user_weights = load_features(user, ftype='user')
            item_weights = load_features(item, ftype='item')

            if user_weights is None or len(user_weights) < LATENT_DIM:
                user_weights = random_weights(LATENT_DIM, scale=INITIAL_THETA_SCALE)

            if item_weights is None or len(item_weights) < LATENT_DIM:
                item_weights = random_weights(LATENT_DIM, scale=INITIAL_THETA_SCALE)

            self.pool.collect(Drop(user, weights=user_weights, ftype='user'),
                              Drop(item, weights=item_weights, ftype='item'),
                              rating=r)

    def on_split_training_data(self) -> list:
        return self.split_by_cpu_count(self.pool.drops)

    @staticmethod
    def split_by_cpu_count(data):
        total_size = len(data)
        size = total_size // CPU_COUNT + 1
        split_list = []
        for i in range(CPU_COUNT):
            start = i * size
            end = min(start + size, total_size)
            split_list.append(data[start:end])
        return split_list

    def on_training(self, drops):
        for epoch in range(OFFLINE_TRAINING_EPOCHS):
            losses = []
            for drop in drops:
                adj_drops, adj_attrs = self.pool.sample_adjacent_drops(drop, k=32)
                x = np.array([adj_drop.weights for adj_drop in adj_drops])
                y = np.array([adj_attr['rating'] for adj_attr in adj_attrs])
                cf_model = CFModel(theta=drop.weights,
                                   latent_dim=LATENT_DIM,
                                   initial_theta_scale=INITIAL_THETA_SCALE)
                cf_model.fit(x, y, epochs=OFFLINE_SAMPLE_TRAINING_EPOCHS)
                drop.weights = cf_model.theta
                losses.append(cf_model.losses[-1])

            self.log_info('Mean loss = {}'.format(sum(losses) / len(losses)), color='blue')

    def on_training_end(self):
        for drop in self.pool.drops:
            save_features(drop.name, ftype=drop.ftype, values=drop.weights)

    def on_split_producing_data(self) -> list:
        users = Features.objects(type='user')
        return self.split_by_cpu_count(users)

    def on_producing(self, users):
        items = Features.objects(type='item')
        x, key_index = features2matrix(items)
        k = 100
        for user in users:
            cf_model = CFModel(theta=user.values)
            predictions = cf_model.predict(x)
            item_preds = sorted(zip(key_index, predictions), key=lambda x: x[1], reverse=True)
            adj_drops, _ = self.pool.get_adjacent_drops(user.key)
            associated_set = set([drop.name for drop in adj_drops])
            limit = k + len(adj_drops)
            user_top_limit_item_set = set([item_pred[0] for item_pred in item_preds[:limit]])
            user_top_k_item_set = (user_top_limit_item_set - associated_set)
            user_top_k_items = filter(lambda kv: kv[0] in user_top_k_item_set, item_preds[:limit])

            # update database
            UserTopKItems.objects(user=user.key).update(items=[k for k, v in user_top_k_items], upsert=True)


if __name__ == '__main__':
    RsPipeline().run()
