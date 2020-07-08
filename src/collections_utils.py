# for python2 and python3 compatibility
try:
    import queue
except ImportError:
    import Queue as queue


class DiscardQueue(queue.Queue, object):
    def __init__(self, size):
        super(DiscardQueue, self).__init__(size)

    def put(self, item):
        if self.qsize() + 1 >= self.maxsize:
            super(DiscardQueue, self).get()  # remove from front
        super(DiscardQueue, self).put(item)


class DiscardList(list):
    def __init__(self, size):
        self.size = size

    def append(self, item):
        if self.__len__() >= self.size:
            super(DiscardList, self).pop(0)
        super(DiscardList, self).append(item)

    def get_avg(self):
        if len(self) <= 1:
            return 1
        return (self[-1] - self[0])/(len(self) - 1) 
