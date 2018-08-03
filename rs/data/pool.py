# Created by wangzixin at 02/08/2018
import networkx as nx
import random


class Pool(object):
    """"""

    def __init__(self):
        """Constructor for Pool"""
        self._graph = nx.Graph()
        self._lookup_table = dict()

    @property
    def drops(self):
        return list(self._graph.nodes)

    def collect(self, drop_a, drop_b, **kwargs):

        str_drop_a = str(drop_a)
        str_drop_b = str(drop_a)
        if str_drop_a not in self._lookup_table:
            self._lookup_table[str_drop_a] = drop_a

        if str_drop_b not in self._lookup_table:
            self._lookup_table[str_drop_b] = drop_b

        self._graph.add_edge(drop_a, drop_b, **kwargs)

    def _get_drops_edges(self, drop, drops):
        attrs = [self._graph.edges[drop, adj_drop] for adj_drop in drops]
        return attrs

    def _str2drop(self, drop):
        if isinstance(drop, str):
            drop = self._lookup_table[drop]
        return drop

    def get_adjacent_drops(self, drop) -> (list, list):
        drop = self._str2drop(drop)
        drops = list(self._graph[drop])
        attrs = self._get_drops_edges(drop, drops)
        return drops, attrs

    def sample_adjacent_drops(self, drop, k) -> (list, list):
        drop = self._str2drop(drop)
        all_adj_drops = list(self._graph[drop])
        if len(all_adj_drops) < k:
            random.shuffle(all_adj_drops)
            drops = all_adj_drops
        else:
            drops = random.sample(all_adj_drops, k)
        edges = self._get_drops_edges(drop, drops)
        return drops, edges

    def empty(self):
        self._graph.clear()
        self._lookup_table.clear()


class Drop(object):
    """"""

    def __init__(self, name, **kwargs):
        """Constructor for Drop"""
        self.name = name
        for k, arg in kwargs.items():
            self.__setattr__(k, arg)

    def __str__(self):
        return self.name
