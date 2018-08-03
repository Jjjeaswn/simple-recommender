# Created by wangzixin at 02/08/2018

from threading import Thread
from datetime import datetime
from abc import abstractmethod, ABC
import multiprocessing

NB_CPU = multiprocessing.cpu_count()


class Pipeline(ABC):
    """"""

    def __init__(self):
        """"""
        self._pre_log_time = None
        self._logs = []

    @abstractmethod
    def on_loading(self):
        pass

    @abstractmethod
    def on_training(self, data):
        pass

    @abstractmethod
    def on_training_end(self, ):
        pass

    @abstractmethod
    def on_producing(self, data):
        pass

    @abstractmethod
    def on_split_training_data(self) -> list:
        pass

    @abstractmethod
    def on_split_producing_data(self) -> list:
        pass

    def log(self, message: str):
        now = datetime.now()
        if self._pre_log_time is not None:
            self._log('Pipeline - {} : previous task consumed {} '.format(now, now - self._pre_log_time))

        self._log('Pipeline - {} : {}'.format(now, message))
        self._pre_log_time = now

    def _log(self, content):
        self._logs.append(content)
        print(content)

    def run(self):
        self.log('Pipeline starts ...')

        self.log('On loading begins ...')
        self.on_loading()
        self.log('On loading ended ...')

        self.log('On training begins ...')
        self.__split_process(self.on_split_training_data, self.on_training)
        self.on_training_end()
        self.log('On training ended ..')

        self.log('On producing begins ..')
        self.__split_process(self.on_split_producing_data, self.on_producing)
        self.log('On producing ended ..')

    def __split_process(self, split_func, process_func):
        data_list = split_func()
        data_len = len(data_list)

        if data_len > NB_CPU:
            print('Warning: number of threads lager than number of CPUs ...')

        if data_len > 1:
            threads = []
            for data in data_list:
                t = Thread(target=process_func, args=(data,))
                t.start()
                threads.append(t)

            for thread in threads:
                thread.join()

        elif data_len == 1:
            process_func(self, data_list[0])
