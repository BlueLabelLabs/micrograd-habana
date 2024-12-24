from abc import ABC, abstractmethod


class Backend(ABC):
    @staticmethod
    @abstractmethod
    def add(a, b):
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def mul(a, b):
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def pow(a, b):
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def relu(a):
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def backward(a, b):
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def neq(a, b):
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def radd(a, b):
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def sub(a, b):
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def rsub(a, b):
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def rmul(a, b):
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def truediv(a, b):
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def rtruediv(a, b):
        raise NotImplementedError
