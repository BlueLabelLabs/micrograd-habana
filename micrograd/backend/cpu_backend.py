from micrograd.backend import Backend


class CPUBackend(Backend):
    @staticmethod
    def add(a, b):
        return a + b

    @staticmethod
    def mul(a, b):
        return a * b

    @staticmethod
    def pow(a, b):
        return a**b

    @staticmethod
    def neq(a, b):
        return a != b

    @staticmethod
    def radd(a, b):
        return b + a

    @staticmethod
    def sub(a, b):
        return a - b

    @staticmethod
    def rsub(a, b):
        return b - a

    @staticmethod
    def rmul(a, b):
        return b * a

    @staticmethod
    def truediv(a, b):
        return a / b

    @staticmethod
    def rtruediv(a, b):
        return b / a
