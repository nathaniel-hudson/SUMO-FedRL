import json
import os.path as path

from typing import Any, Dict


class Counter:

    data: Dict[Any, int]
    __META_COUNTER_PATH = path.expanduser(path.join("~", ".sumo-fedrl-counter.json"))
    
    def __init__(self) -> None:
        if not path.exists(self.path):
            self.data = dict()
        else:
            with open(self.path, "r") as json_file:
                self.data = json.load(json_file)

    def get(self, key) -> int:
        if key not in self.data:
            self.data[key] = 0
            self.__save()
        return self.data[key]
        
    def increment(self, key) -> None:
        if key in self.data:
            self.data[key] += 1
        else:
            self.data[key] = 1
        self.__save()

    def __save(self) -> None:
        with open(self.path, "w") as json_file:
            json.dump(self.data, json_file)

    @property
    def path(self) -> str:
        return Counter.__META_COUNTER_PATH