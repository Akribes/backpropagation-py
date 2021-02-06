import math

weights = [  # TODO reverse from and to, so I can remove bias weights easily using [:-1]
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


def C(y, x):
    res = []
    for i in range(len(y)):
        res.append(y[i] - x[i])
    return res


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
    if len(A[0]) != len(B):
        raise ValueError("Matrix sizes do not match")
    res = [[0] * len(B[0]) for i in range(len(A))]

    for j in range(len(res)):
        for i in range(len(res[j])):
            for k in range(len(B)):
                res[j][i] += A[j][k] * B[k][i]
    return res


def transpose_matrix(A):
    AT = [[0] * len(A) for i in range(len(A[0]))]

    for j in range(len(A)):
        for i in range(len(A[j])):
            AT[i][j] = A[j][i]

    return AT


def propagate(x):
    for i in range(len(weights)):
        if i == 0:
            o = x[:]  # TODO set these values as the activation of the first layer
        else:
            o = activation[i - 1]
        o.append(1)  # bias

        z = transpose_matrix([o])
        z = multiply_matrices(weights[i], z)
        activation[i] = f_vector(z)
        derivatives[i] = f_deriv_vector(z)

    return activation[-1]  # return output


def backpropagate(y):
    errors = [
        [],
        []
    ]

    errors[1] = C(y, activation[-1])

    i = 0
    while i >= 0:
        errors[i] = multiply_matrices(derivatives[i], transpose_matrix(weights[i + 1][:-1]))  # [:-1] removes bias weight
        print("Errors for layer", i, "are", errors[i])
        i -= 1


def main():
    data = [
        [[0, 0], [0]],
        [[0, 1], [1]],
        [[1, 0], [1]],
        [[1, 1], [0]]
    ]
    print(propagate(data[0][0]), activation, derivatives)
    backpropagate(data[0][1])


if __name__ == "__main__":
    main()
