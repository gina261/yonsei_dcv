import numpy as np
from grad import numeric_backward, numeric_gradient
from two_layer_net import Linear, ReLU, softmax_loss, l2_regularization, \
                          TwoLayerNet


def check_softmax_stability():
    """ A simple test-case to check the numeric stability of softmax_loss """
    big = 1e3
    x = np.array([
          [big, 0, -big],  # noqa: E126
          [-big, big, 0],  # noqa: E126
          [0, -big, big],  # noqa: E126
        ])                 # noqa: E126
    y = np.array([0, 1, 2])
    loss, _ = softmax_loss(x, y)
    if loss is None:
        print('You have not yet implemented softmax_loss')
        return
    print('Input scores:')
    print(x)
    print(f'Input labels: {y}')
    print(f'Output loss: {loss}')
    if np.isnan(loss):
        print('Your softmax_loss gave a NaN with big input values.')
        print('Did you forget to implement the max-subtraction trick?')
    else:
        print('It seems like your softmax_loss is numerically stable!')


def gradcheck_linear():
    print('Running numeric gradient check for linear')
    N, Din, Dout = 3, 4, 5
    x = np.random.randn(N, Din)
    w = np.random.randn(Din, Dout)
    b = np.random.randn(Dout)

    y, cache = Linear.forward(x, w, b)
    if y is None:
        print('  Forward pass is not implemented!')
        return

    dy = np.random.randn(*y.shape)
    dx, dw, db = Linear.backward(dy, cache)
    if dx is None or dw is None or db is None:
        print('  Backward pass is not implemented!')
        return

    fx = lambda _: Linear.forward(_, w, b)[0]
    dx_numeric = numeric_backward(fx, x, dy)
    max_diff = np.abs(dx - dx_numeric).max()
    print('  dx difference: ', max_diff)

    fw = lambda _: Linear.forward(x, _, b)[0]
    dw_numeric = numeric_backward(fw, w, dy)
    max_diff = np.abs(dw - dw_numeric).max()
    print('  dw difference: ', max_diff)

    fb = lambda _: Linear.forward(x, w, _)[0]
    db_numeric = numeric_backward(fb, b, dy)
    max_diff = np.abs(db - db_numeric).max()
    print('  db difference: ', max_diff)


def gradcheck_relu():
    print('Running numeric gradient check for relu')
    N, Din = 4, 5
    x = np.random.randn(N, Din)

    y, cache = ReLU.forward(x)
    if y is None:
        print('  Forward pass is not implemented!')
        return

    dy = np.random.randn(*y.shape)
    dx = ReLU.backward(dy, cache)
    if dx is None:
        print('  Backward pass is not implemented!')
        return

    f = lambda _: ReLU.forward(_)[0]
    dx_numeric = numeric_backward(f, x, dy)
    max_diff = np.abs(dx - dx_numeric).max()
    print('  dx difference: ', max_diff)


def gradcheck_softmax():
    print('Running numeric gradient check for softmax loss')
    N, C = 4, 5
    x = np.random.randn(N, C)
    y = np.random.randint(C, size=(N,))
    loss, dx = softmax_loss(x, y)
    if loss is None or dx is None:
        print('  Softmax not implemented!')
        return

    f = lambda _: softmax_loss(_, y)[0]
    dx_numeric = numeric_gradient(f, x)
    max_diff = np.abs(dx - dx_numeric).max()
    print('  dx difference: ', max_diff)


def gradcheck_l2_regularization():
    print('Running numeric gradient check for L2 regularization')
    Din, Dout = 3, 4
    reg = 0.1
    w = np.random.randn(Din, Dout)
    loss, dw = l2_regularization(w, reg)
    if loss is None or dw is None:
        print('  L2 regularization not implemented!')
        return

    f = lambda _: l2_regularization(_, reg)[0]
    dw_numeric = numeric_gradient(f, w)
    max_diff = np.abs(dw - dw_numeric).max()
    print('  dw difference: ', max_diff)


def gradcheck_two_layer_net():
    print('Running numeric gradient check for TwoLayerNet')
    N, D, C, H = 3, 4, 5, 6
    model = TwoLayerNet(input_dim=D, num_classes=C, hidden_dim=H)

    X = np.random.randn(N, D)
    y = np.random.randint(C, size=(N,))
    dscores = np.random.randn(N, C)

    _, dscores = softmax_loss(model.loss(X), y)
    _, grads = model.loss(X, y)

    numeric_grads = {}
    for k, param in model.parameters().items():
        def f(_):
            old_val = param.copy()
            param[:] = _
            scores = model.loss(X)
            param[:] = old_val
            return scores
        numeric_grads[k] = numeric_backward(f, param, dscores)


    assert grads.keys() == numeric_grads.keys()
    for k, grad in grads.items():
        numeric_grad = numeric_grads[k]
        max_diff = np.abs(grad - numeric_grad).max()
        print(f'  Max diff for d{k}: ', max_diff)


def main():
    gradcheck_linear()
    print()
    gradcheck_relu()
    print()
    gradcheck_softmax()
    print()
    check_softmax_stability()
    print()
    gradcheck_l2_regularization()
    print()
    gradcheck_two_layer_net()
    print()


if __name__ == '__main__':
    main()
