from pygrad import Scalar


def main():
    # Inputs
    x1 = Scalar(2.0)
    x2 = Scalar(0.0)

    # Weights
    w1 = Scalar(-3.0)
    w2 = Scalar(1.0)

    # Bias
    b = Scalar(6.8813735870195432)

    x1w1 = x1*w1
    x2w2 = x2*w2

    x1w1x2xw2 = x1w1 + x2w2

    n = x1w1x2xw2 + b

    e = (2*n).exp()
    o = (e - 1) / (e + 1)

    o.backward()


if __name__ == "__main__":
    main()
