from abc import ABC
from pandas import DataFrame


class BaseTester(ABC):

    def __init__(
        self
    ) -> None:
        pass

    def test() -> DataFrame:
        pass

    def load_model(checkpoint: str=None) -> None:
        pass
