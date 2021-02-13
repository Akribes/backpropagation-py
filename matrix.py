class Matrix:
    def __init__(self, *args):
        if len(args) == 1:
            self.data = args[0]
        else:
            self.data = [[0 for _ in range(args[1])] for _ in range(args[0])]

    def __str__(self):
        return "{} * {} matrix {}".format(self.m(), self.n(), str(self.data))

    def __mul__(self, other):
        if self.n() != other.m():
            raise ValueError("Matrix sizes do not match: multiplying {} and {}".format(self, other))
        res = Matrix(self.m(), other.n())

        for j in range(res.m()):
            for i in range(res.n()):
                for k in range(self.n()):
                    res.data[j][i] += self.data[j][k] * other.data[k][i]
        return res

    def __add__(self, other):
        if self.m() != other.m() or self.n() != other.n():
            raise ValueError("Matrix sizes do not match:", self, other)
        res = Matrix(self.m(), self.n())
        for i in range(self.m()):
            for j in range(self.n()):
                res.data[i][j] = self.data[i][j] + other.data[i][j]
        return res

    def __sub__(self, other):
        if self.m() != other.m() or self.n() != other.n():
            raise ValueError("Matrix sizes do not match:", self, other)
        res = Matrix(self.m(), self.n())
        for i in range(self.m()):
            for j in range(self.n()):
                res.data[i][j] = self.data[i][j] - other.data[i][j]
        return res

    def m(self):
        return len(self.data)

    def n(self):
        return len(self.data[0])

    def transpose(self):
        t = Matrix(self.n(), self.m())
        for j in range(self.m()):
            for i in range(self.n()):
                t.data[i][j] = self.data[j][i]
        return t

    def hadamard_mul(self, other):
        res = Matrix(self.data)
        for i in range(res.m()):
            for j in range(res.n()):
                res.data[i][j] *= other.data[i][j]
        return res

    def scalar_mul(self, x):
        res = Matrix(self.data)
        for i in range(res.m()):
            for j in range(res.n()):
                res.data[i][j] *= x
        return res

    def scalar_div(self, x):
        return self.scalar_mul(1 / x)

    def __getitem__(self, item):
        if isinstance(item[0], slice):
            res = self.data[item[0]]
        else:
            res = [self.data[item[0]]]

        for i in range(len(res)):
            if isinstance(item[1], slice):
                res[i] = res[i][item[1]]
            else:
                res[i] = [res[i][item[1]]]
        if not isinstance(item[0], slice) and not isinstance(item[1], slice):
            return res
        else:
            return Matrix(res)

    def __setitem__(self, key, value):
        self.data[key[0] - 1][key[1] - 1] = value

    def __delitem__(self, key):
        self.data[key[0] - 1][key[1] - 1] = 0
