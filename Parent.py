import copy

class Parent:

    def __init__(self):
        self._c1 = None
        self._c2 = None

    def get_c1(self):
        return self._c1

    def set_c1(self, c1):
        self._c1 = copy.deepcopy(c1)

    def get_c2(self):
        return self._c2

    def set_c2(self, c2):
        self._c2 = copy.deepcopy(c2)

    def set_parent(self, parent):
        if self._c1 is None:
            self.set_c1(parent)
        else:
            self.set_c2(parent)
