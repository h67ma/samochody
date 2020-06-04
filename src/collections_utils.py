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
