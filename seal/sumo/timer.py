import numpy as np

from seal.sumo.config import MAX_DELAY, MIN_DELAY

'''
TODO
We might need to adjust the ActionTimer class such that it considers
the direct traffic light states when deciding the delay value (i.e., traffic light
states might have varying delays).

TODO
We need to consider a MAXIMUM amount of time that's allowed to pass before a traffic light
MUST change.
'''


class ActionTimer:
    """Class implements the necessary "timer" mechanism in order to detect whether a
       trafficlight has sat idle long enough until it can change its light phase.
    """

    def __init__(self, n_actions: int, delay: int=MIN_DELAY):
        self.__n_actions = n_actions
        self.delay = delay
        self.restart()

    def restart(self, index: int=None) -> None:
        if index is not None:
            self.__timer[index] = 0.0
        else:
            self.__timer = np.zeros(shape=(self.__n_actions))

    def incr(self, index: int=None) -> None:
        if index is not None:
            self.__timer[index] += 1
        else:
            for i in range(self.__n_actions):
                self.__timer[i] += 1

    def can_change(self, index: int) -> bool:
        return self.__timer[index] >= MIN_DELAY

    def must_change(self, index: int) -> bool:
        return self.__timer[index] > MAX_DELAY

    def __repr__(self) -> str:
        return str(self.__timer)
