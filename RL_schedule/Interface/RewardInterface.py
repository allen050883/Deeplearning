import abc


class Reward(abc.ABC):
    @abc.abstractmethod
    def get_episode_ended_reward(self, state):
        pass

    @abc.abstractmethod
    def get_episode_not_ended_reward(self, state):
        pass
