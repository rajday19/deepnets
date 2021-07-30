import torch
from torch import tensor
import time as time


def matmul(a, b):
    ar, ac = a.shape
    br, bc = b.shape
    assert ac == br
    c = torch.zeros(ar, bc)
    for i in range(ar):
        for j in range(bc):
            for k in range(ac):
                c[i, j] += a[i, k]*b[k, j]
    return c


def matmul1x(a, b):
    ar, ac = a.shape
    br, bc = b.shape
    assert ac == br
    c = torch.zeros(ar, bc)
    for i in range(ar):
        for j in range(bc):
            c[i, j] = (a[i, :] * b[:, j]).sum()
    return c


def matmul2x(a, b):
    ar, ac = a.shape
    br, bc = b.shape
    assert ac == br
    c = torch.zeros(ar, bc)
    for i in range(ar):
        c[i] = (a[i].unsqueeze(-1) * b).sum(dim=0)
    return c


if __name__ == "__main__":
    m1 = torch.randn(5, 28*28)
    m2 = torch.randn(784, 10)

    start = time.time()
    t1 = matmul(m1, m2)
    end = time.time()
    print("Time taken for matmul is: ", (end-start))

    start = time.time()
    t1 = matmul1x(m1, m2)
    end = time.time()
    print("Time taken for matmul1x is: ", (end-start))

    start = time.time()
    t2 = matmul2x(m1, m2)
    end = time.time()
    print("Time taken for matmul1x is: ", (end-start))

    print("\nTime for broadcasting")
    m = tensor([[-10., -4, -1], [2, 3, 2]])
    print(m**2)
    print(m.shape)
