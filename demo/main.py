from pygrad.engine import Scalar
from pygrad.nn import MLP


def main():
    model = MLP(3, [4, 4, 1])

    xs = [[2.0,  3.0, -1.0],
          [3.0, -1.0,  0.5],
          [0.5,  1.0,  1.0],
          [1.0,  1.0, -1.0]]
    ys = [1.0, -1.0, -1.0, 1.0]

    for k in range(50):
        # Forward Pass
        ypred = [model(x) for x in xs]
        loss = sum((yout - ygt)**2 for ygt, yout in zip(ys, ypred))

        # Backward Pass
        for p in model.parameters():
            p.grad = 0.0
        loss.backward()

        # Update
        for p in model.parameters():
            p.value += -0.05 * p.grad

        print(k, loss.value)

    print([y.value for y in ypred])


if __name__ == "__main__":
    main()
