class Queue(list):
    def __init__(self, size):
        super().__init__()
        self.size = size

    def append(self, item):
        if self.__len__() >= self.size:
            self.pop(0)
        super(Queue, self).append(item)