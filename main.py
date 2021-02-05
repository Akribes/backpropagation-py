import math

weights = [
    [
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0]
    ],
    [
        [0, 0, 0, 0]
    ]
]

activation = [
    [],
    []
]

derivatives = [
    [],
    []
]

loss = 0


def C(y, x):
    return y - x


def f(x):
    return 1 / (1 + math.exp(-x))


def f_deriv(x):
    return f(x) * f(-x)


def f_vector(x):
    res = []
    for a in x:
        res.append(f(a[0]))
    return res


def f_deriv_vector(x):
    res = []
    for a in x:
        res.append(f_deriv(a[0]))
    return res


def multiply_matrices(A, B):
    print("Multiplying", A, "and", B)
    if len(A[0]) != len(B):
        raise ValueError("Matrix sizes do not match")
    res = [[0] * len(B[0]) for i in range(len(A))]

    for j in range(len(res)):
        for i in range(len(res[j])):
            for k in range(len(B)):
                res[j][i] += A[j][k] * B[k][i]
    return res


def transpose_matrix(A):
    print("Transposing", A)
    AT = [[0] * len(A) for i in range(len(A[0]))]

    for j in range(len(A)):
        for i in range(len(A[j])):
            AT[i][j] = A[j][i]

    return AT


def propagate(x):
    for i in range(len(weights)):
        o = None
        if i == 0:
            o = x[:]
            print(o)
        else:
            o = activation[i - 1]
        o.append(1)  # bias

        z = transpose_matrix([o])
        z = multiply_matrices(weights[i], z)
        print("Weighted inputs", z)
        activation[i] = f_vector(z)
        derivatives[i] = f_deriv_vector(z)

    return activation[-1]  # return output


print(propagate([1, 2]))
