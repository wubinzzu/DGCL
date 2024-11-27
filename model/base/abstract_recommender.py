__author__ = "Zhongchuan Sun"
__email__ = "zhongchuansun@gmail.com"

__all__ = ["AbstractRecommender"]

from reckit import Logger
from reckit import Configurator
# from reckit import Evaluator
from evaluate.evaluator import Evaluator
from reckit import typeassert
from data import Dataset
import pandas as pd
import numpy as np
import scipy.sparse as sp
import abc
import time
import os


@typeassert(config=Configurator, data_name=str)
def _create_logger(config, data_name):
    # create a logger
    timestamp = time.time()
    model_name = config.recommender
    param_str = f"{data_name}_{model_name}_{config.summarize()}"
    run_id = f"{param_str[:150]}_{timestamp:.8f}"

    log_dir = os.path.join("log", data_name, model_name)
    logger_name = os.path.join(log_dir, run_id + ".log")
    logger = Logger(logger_name)

    return logger


class AbstractRecommender(object):
    @typeassert(config=Configurator)
    def __init__(self, config):
        self.dataset = Dataset(config.data_dir, config.sep, config.file_column)
        self.logger = self._create_logger(config, self.dataset)

        user_train_dict = self.dataset.train_data.to_user_dict()
        user_test_dict = self.dataset.test_data.to_user_dict()
        self.evaluator = Evaluator(user_train_dict, user_test_dict,
                                   metric=config.metric, top_k=config.top_k,
                                   batch_size=config.test_batch_size,
                                   num_thread=config.test_thread,
                                   group_view=config.group_view)

    @typeassert(config=Configurator, dataset=Dataset)
    def _create_logger(self, config, dataset):
        timestamp = time.time()
        if "pytorch" in self.__class__.__module__:
            model_name = "torch_" + self.__class__.__name__
        elif "tensorflow" in self.__class__.__module__:
            model_name = "tf_" + self.__class__.__name__
        else:
            model_name = self.__class__.__name__
        data_name = dataset.data_name
        param_str = f"{data_name}_{model_name}_{config.summarize()}"
        run_id = f"{param_str[:150]}_{timestamp:.8f}"

        log_dir = os.path.join("log", data_name, self.__class__.__name__)
        logger_name = os.path.join(log_dir, run_id + ".log")
        logger = Logger(logger_name)

        logger.info(f"my pid: {os.getpid()}")
        logger.info(f"model: {self.__class__.__module__}")
        logger.info(self.dataset)
        logger.info(config)

        return logger

    @abc.abstractmethod
    def train_model(self):
        pass

    @abc.abstractmethod
    def predict(self, users):
        pass

class SocialAbstractRecommender(AbstractRecommender):
    @typeassert(config=Configurator)
    def __init__(self, config):
        # super(SocialAbstractRecommender, self).__init__(config)
        # file_prefix = os.path.join(config.data_dir, os.path.split(config.data_dir)[-1])
        # self.social_users = pd.read_csv(file_prefix+".trust", sep=config.sep, header=None, names=["user", "friend"])
        # users_key = np.array(list(self.dataset.user2id.keys()))
        # index = np.in1d(self.social_users["user"], users_key)
        # social_users = self.social_users[index]
        #
        # index = np.in1d(social_users["friend"], users_key)
        # social_users = social_users[index]
        #
        # user = social_users["user"]
        # user_id = [self.dataset.user2id[u] for u in user]
        # friend = social_users["friend"]
        # friend_id = [self.dataset.user2id[u] for u in friend]
        # num_users = self.dataset.train_data.num_users
        # self.social_matrix = sp.csr_matrix(([1] * len(user_id), (user_id, friend_id)),
        #                                    shape=(num_users, num_users

        super(SocialAbstractRecommender, self).__init__(config)
        self.social_users = pd.read_csv(config["social_file"], sep='\t',
                                        header=None, names=["uid", "fid"])
        users_key = np.array(list(self.dataset.user2id.keys()))
        index = np.in1d(self.social_users["uid"], users_key)
        social_users = self.social_users[index]

        index = np.in1d(social_users["fid"], users_key)
        social_users = social_users[index]

        user = social_users["uid"]
        user_id = [self.dataset.user2id[u] for u in user]
        friend = social_users["fid"]
        friend_id = [self.dataset.user2id[u] for u in friend]
        num_users = self.dataset.train_data.num_users
        self.social_matrix = sp.csr_matrix(([1] * len(user_id), (user_id, friend_id)),
                                           shape=(num_users, num_users))