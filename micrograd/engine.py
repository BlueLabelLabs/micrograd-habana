import os

from micrograd.backend import CPUBackend, HPUBackend
from micrograd.utils import get_device_initial


class Value:
    """stores a single scalar value and its gradient"""

    def __init__(self, data, _children=(), _op=""):
        self.data = data
        self.grad = 0
        # internal variables used for autograd graph construction
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op  # the op that produced this node, for graphviz / debugging / etc

        # 1. Determine the device from an environment variable or default
        env_preferred_device = os.environ.get("PREFERRED_DEVICE", None)
        final_device = get_device_initial(env_preferred_device)

        # 2. Instantiate the correct backend (no self.device, only self.backend)
        if final_device == "hpu":
            self.backend = HPUBackend()
        # # As for now do not add GPUBackend
        # elif final_device == "cuda":
        #     self.backend = GPUBackend()
        else:
            # Could add CUDABackend if needed; defaulting to CPU here
            self.backend = CPUBackend()

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out_data = self.backend.add(self.data, other.data)
        out = Value(out_data, (self, other), "+")

        def _backward():
            self.grad += out.grad
            other.grad += out.grad

        out._backward = _backward

        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out_data = self.backend.mul(self.data, other.data)
        out = Value(out_data, (self, other), "*")

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward

        return out

    def __pow__(self, other):
        assert isinstance(
            other, (int, float)
        ), "only supporting int/float powers for now"
        out_data = self.backend.mul(self.data, other.data)
        out = Value(out_data, (self,), f"**{other}")

        def _backward():
            self.grad += (other * self.data ** (other - 1)) * out.grad

        out._backward = _backward

        return out

    def relu(self):
        out = Value(0 if self.data < 0 else self.data, (self,), "ReLU")

        def _backward():
            self.grad += (out.data > 0) * out.grad

        out._backward = _backward

        return out

    def backward(self):
        # topological order all of the children in the graph
        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)

        build_topo(self)

        # go one variable at a time and apply the chain rule to get its gradient
        self.grad = 1
        for v in reversed(topo):
            v._backward()

    def __neg__(self):  # -self
        return self.backend.neg(self)

    def __radd__(self, other):  # other + self
        return self.backend.add(other, self)

    def __sub__(self, other):  # self - other
        return self.backend.sub(self, other)

    def __rsub__(self, other):  # other - self
        return self.backend.sub(other, self)

    def __rmul__(self, other):  # other * self
        return self.backend.mul(other, self)

    def __truediv__(self, other):  # self / other
        return self.backend.truediv(self, other)

    def __rtruediv__(self, other):  # other / self
        return self.backend.truediv(other, self)

    def __repr__(self):
        return f"Value(data={self.data}, grad={self.grad})"
