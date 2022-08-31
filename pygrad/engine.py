import math
from enum import Enum


class Op(Enum):
    ADD = 'ADD'
    MUL = 'MUL'
    POW = 'POW'
    EXP = 'EXP'
    TANH = 'TANH'


class Scalar:
    def __init__(self, value, parents=(), op=''):
        self.value = value
        self.grad = 0
        self.parents = set(parents)
        self.op = op

        self._backward = lambda: None

    def __repr__(self):
        return f"Scalar(value={self.value}, grad={self.grad})"

    def __add__(self, other):
        other = other if isinstance(other, Scalar) else Scalar(other)
        out = Scalar(self.value + other.value, (self, other), Op.ADD)

        def _backward():
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad
        out._backward = _backward

        return out

    def __radd__(self, other):
        # other + self
        return self + other

    def __sub__(self, other):
        # self - other
        return self + (-other)

    def __rsub__(self, other):
        # other - self
        return -self + other

    def __neg__(self):
        # -self
        return self * -1

    def __mul__(self, other):
        other = other if isinstance(other, Scalar) else Scalar(other)
        out = Scalar(self.value * other.value, (self, other), Op.MUL)

        def _backward():
            self.grad += other.value * out.grad
            other.grad += self.value * out.grad
        out._backward = _backward

        return out

    def __rmul__(self, other):
        # other * self
        return self * other

    def __truediv__(self, other):
        # self / other
        return self * other**-1

    def __pow__(self, other):
        assert isinstance(other, (int, float))
        out = Scalar(self.value**other, (self,), Op.POW)

        def _backward():
            self.grad += (other * self.value**(other - 1)) * out.grad
        out._backward = _backward

        return out

    def exp(self):
        x = self.value
        out = Scalar(math.exp(x), (self,), Op.EXP)

        def _backward():
            self.grad += out.value * out.grad
        out._backward = _backward

        return out

    def tanh(self):
        x = self.value
        t = (math.exp(2*x) - 1) / (math.exp(2*x) + 1)
        out = Scalar(t, (self,), Op.TANH)

        def _backward():
            self.grad += (1 - t**2) * out.grad
        out._backward = _backward

        return out

    def backward(self):
        ordered = []
        visited = set()

        # Topological Sort
        def sort(v):
            if v not in visited:
                visited.add(v)
                for child in v.parents:
                    sort(child)
                ordered.append(v)
        sort(self)

        self.grad = 1.0
        for node in reversed(ordered):
            node._backward()
