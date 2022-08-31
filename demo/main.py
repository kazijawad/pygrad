from pygrad.engine import Scalar
from pygrad.nn import MLP


def main():
    x = [2.0, 3.0, -1.0]
    model = MLP(3, [4, 4, 1])
    out = model(x)
    print(out)


if __name__ == "__main__":
    main()
