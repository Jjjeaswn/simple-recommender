# Created by wangzixin at 02/08/2018

from threading import Thread
from datetime import datetime
from abc import abstractmethod, ABC
import multiprocessing

NB_CPU = multiprocessing.cpu_count()


class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'

    @classmethod
    def disable(cls):
        cls.HEADER = ''
        cls.OKBLUE = ''
        cls.OKGREEN = ''
        cls.WARNING = ''
        cls.FAIL = ''
        cls.ENDC = ''


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

    def log_info(self, message, color=None):
        header = 'Pipeline - {} : '.format(datetime.now())
        self._log(header, message, color=color)

    def log_step_info(self, message: str, color=None):
        now = datetime.now()
        header = 'Pipeline - {} : '.format(now)
        if self._pre_log_time is not None:
            self._log(header, 'previous task consumed {} '.format(now - self._pre_log_time),
                      color='green')

        self._log(header, message, color=color)
        self._pre_log_time = now

    def _log(self, header, content, color=None):
        if color == 'green':
            content = ''.join([Colors.HEADER, header, Colors.OKGREEN, content, Colors.ENDC])
        elif color == 'red':
            content = ''.join([Colors.HEADER, header, Colors.WARNING, content, Colors.ENDC])
        elif color == 'blue':
            content = ''.join([Colors.HEADER, header, Colors.OKBLUE, content, Colors.ENDC])
        else:
            content = ''.join([Colors.HEADER, header, Colors.ENDC, content])

        self._logs.append(content)
        print(content)

    def run(self):
        self.log_step_info('Pipeline starts ...', color='blue')
        self.log_step_info('-' * 50)

        self.log_step_info('On loading begins ...', color='blue')
        self.on_loading()
        self.log_step_info('On loading ended ...', color='blue')
        self.log_step_info('-' * 50)

        self.log_step_info('On training begins ...', color='blue')
        self.__split_process(self.on_split_training_data, self.on_training)
        self.on_training_end()
        self.log_step_info('On training ended ..', color='blue')
        self.log_step_info('-' * 50)

        self.log_step_info('On producing begins ..', color='blue')
        self.__split_process(self.on_split_producing_data, self.on_producing)
        self.log_step_info('On producing ended ..', color='blue')
        self.log_step_info('-' * 50)

        self.log_step_info('Success!', color='green')

    def __split_process(self, split_func, process_func):
        data_list = split_func()
        data_len = len(data_list)

        if data_len > NB_CPU:
            print(Colors.WARNING + 'Warning: number of threads lager than number of CPUs ...' + Colors.ENDC)

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
