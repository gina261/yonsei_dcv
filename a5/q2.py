import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import imageio
import io

from fitting import affine_transform_loss, fit_affine_transform
from grad import numeric_gradient

"""
Main entry point for the fitting question.

Students should not need to modify this file.
"""


parser = argparse.ArgumentParser()
parser.add_argument(
    'action',
    choices=['gradcheck', 'fit'])
parser.add_argument(
    '--learning-rate',
    type=float,
    default=1e-4,
    help='Learning rate to use for gradient descent')
parser.add_argument(
    '--steps',
    type=int,
    default=100,
    help='Number of iterations to use for gradient descent')
parser.add_argument(
    '--data-file',
    default='data/points_case_1.npy',
    help='Path to input data file of correspondences')
parser.add_argument(
    '--print-loss-every',
    type=int,
    default=50,
    help='How frequently to print losses during fitting')
parser.add_argument(
    '--loss-plot',
    default='loss',
    help='If given, save a plot of the losses to this file.')
parser.add_argument(
    '--animated-gif',
    default='fitting',
    help='If given, save an animated GIF of the fitting process to this file.')


def gradcheck(num_trials=100, tolerance=1e-8):
    N = 100
    print('Running numeric gradient checks for affine_transform_loss')
    for _ in range(num_trials):
        X = np.random.randn(N, 2)
        Y = np.random.randn(N, 2)
        S = np.random.randn(2, 2)
        t = np.random.randn(2)
        f_S = lambda _: affine_transform_loss(X, Y, _, t)[0]  # noqa: E731
        f_t = lambda _: affine_transform_loss(X, Y, S, _)[0]  # noqa: E731
        loss, pred, grad_S, grad_t = affine_transform_loss(X, Y, S, t)
        if loss is None:
            print('FAIL: Forward pass not implemented')
            return
        elif grad_S is None or grad_t is None:
            print('FAIL: Backward pass not implemented')
            return
        numeric_grad_S = numeric_gradient(f_S, S)
        numeric_grad_t = numeric_gradient(f_t, t)
        grad_S_max_diff = np.abs(numeric_grad_S - grad_S).max()
        grad_t_max_diff = np.abs(numeric_grad_t - grad_t).max()
        if grad_S_max_diff > tolerance:
            print('FAIL: grad_S not within tolerance')
            print('grad_S:')
            print(grad_S)
            print('numeric_grad_S:')
            print(numeric_grad_S)
            print(f'Max difference: {grad_S_max_diff}')
            return
        if grad_t_max_diff > tolerance:
            print('FAIL: grad_t not within tolerance')
            print(f'grad_t: {grad_t}')
            print(f'grad_t_numeric: {numeric_grad_t}')
            print(f'Max difference: {grad_t_max_diff}')
            return
    print('PASSED')


class Logger:
    def __init__(self, P, P_prime, print_every=50):
        self.P = P
        self.P_prime = P_prime
        self.print_every = print_every
        self.iterations = []
        self.losses = []
        self.predictions = []

    def log(self, iteration, loss, prediction):
        if iteration == 0 or (iteration + 1) % self.print_every == 0:
            print(f'Iteration {iteration}, loss = {loss}')
        self.iterations.append(iteration)
        self.losses.append(loss)
        self.predictions.append(prediction)

    def save_loss_plot(self, filename):
        plt.plot(self.iterations, self.losses, 'o')
        plt.savefig(filename + '.png')
        plt.clf()
        print(f'Saved loss plot to {filename}.png')

    def save_animated_gif(self, filename, show_every=50):
        imgs = []
        factors = (self.iterations, self.losses, self.predictions)
        for i, loss, pred in zip(*factors):
            if not (i == 0 or (i + 1) % show_every == 0):
                continue
            plt.scatter(self.P[:, 0], self.P[:, 1],
                        s=.1, color='b', label='X')
            plt.scatter(self.P_prime[:, 0], self.P_prime[:, 1],
                        s=.1, color='r', label='Y')
            plt.scatter(pred[:, 0], pred[:, 1],
                        s=.1, color='g', label='Y_pred')
            plt.axis([-1., 1.5, -1.5, 1.5])
            plt.tight_layout()
            plt.legend(markerscale=10.)
            plt.title(f'Iteration {i}, loss = {loss}')
            fig = plt.gcf()
            buf = io.BytesIO()
            fig.savefig(buf, format='raw')
            H = int(fig.bbox.bounds[3])
            W = int(fig.bbox.bounds[2])
            buf.seek(0)
            img = np.frombuffer(buf.getvalue(), dtype=np.uint8)
            img = img.reshape(H, W, -1)
            # print(f'Making animated GIF: processing iteration {i}')
            imgs.append(img)
            plt.clf()
        imageio.mimwrite(filename + '.gif', imgs)
        imageio.imwrite(filename + '.png', imgs[-1])
        print(f'Saved animated gif to {filename}.gif')
        print(f'Saved last result to {filename}.png')


def fit(args):
    data = np.load(args.data_file)
    X, Y = data[:, :2], data[:, 2:]
    logger = Logger(X, Y, print_every=args.print_loss_every)

    lr = args.learning_rate
    steps = args.steps
    S, t = fit_affine_transform(X, Y, logger, lr, steps)

    print('Final transform:')
    print('S = ')
    print(S)
    print('t = ')
    print(t)

    if args.loss_plot is not None:
        logger.save_loss_plot(os.path.join('q2_result', args.loss_plot))

    if args.animated_gif is not None:
        logger.save_animated_gif(os.path.join('q2_result', args.animated_gif))


def main(args):
    if args.action == 'gradcheck':
        gradcheck()
    elif args.action == 'fit':
        fit(args)


if __name__ == '__main__':
    os.makedirs('q2_result', exist_ok=True)

    main(parser.parse_args())
