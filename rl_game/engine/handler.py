from abc import abstractmethod, ABC

class PlayerHandler(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def handle_input(self, event, field):
        raise NotImplementedError("You must implement 'handle_input'!!")

    @abstractmethod
    def handle_AI_input(self, event, field):
        raise NotImplementedError("You must implement 'handle_AI_input'!!")


class ItemHandler(ABC):
    @abstractmethod
    def pickup(self, target, field):
        raise NotImplementedError("You must implement 'pickup'!!")