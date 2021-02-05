import math

weights = [
        [
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0] # bias
        ],
        [
            [0],
            [0],
            [0],
            [0] # bias
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
        res.append(f(a))
    return res

def f_deriv_vector(x):
    res = []
    for a in x:
        res.append(f_deriv(a))
    return res


def multiply_matrices(A, B):
    if len(A[0]) != len(B):
        raise ValueError("Matrix sizes do not match")
    res = [[0] * len(B[0])] * len(A)

    for row in range(len(res)):
        for cell in range(len(row)):
            for i in range(len(res)):
                pass # TODO actually implement this
            
    return res


def propagate(x):
    for i in range(len(weights)):
        z = None
        if i == 0:
            z = x
        else:
            z = multiply_matrix(weights[i], activation[i - 1])
        activation[i] = f_vector(z)
        derivatives[i] = f_deriv_vector(z)

    return activation[-1] # return output


print(multiply_matrices([[0,0,0],[0,0,0]], [[0,0],[0,0],[0,0]]))
