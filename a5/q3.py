import argparse
import os
import numpy as np
import matplotlib.pyplot as plt

from data import load_fashion_mnist, DataSampler
from two_layer_net import TwoLayerNet, SGD


parser = argparse.ArgumentParser()
parser.add_argument(
    'action',
    choices=['overfit', 'train', 'test'])
parser.add_argument(
    '--plot-file',
    default='loss',
    help='File where loss and accuracy plot should be saved')
parser.add_argument(
    '--checkpoint',
    default='checkpoint',
    help='File where trained model weights should be saved')
parser.add_argument(
    '--print-every',
    type=int,
    default=100,
    help='How often to print losses during training')


def main(args):
    if args.action == 'test':
        data = load_fashion_mnist(num_train=0, num_val=0, seed=0)
        test_sampler = DataSampler(data['X_test'], data['y_test'], 100)
        checkpoint = os.path.join('q3_result', args.checkpoint + '.npy')
        print(f'Loading model from {checkpoint}')
        model = TwoLayerNet(1, 1, 1)
        model.load(checkpoint)
        test_acc = check_accuracy(model, test_sampler)
        print('Checking accuracy')
        print(f'  Test : {test_acc:.2f}')
        return


    # Model architecture hyperparameters.
    hidden_dim = 400
    weight_scale = 1e-3

    if args.action == 'overfit':
        # How much data to use for training
        num_train = 100

        # Optimization hyperparameters.
        batch_size = 10
        num_epochs = 30
        momentum = 0.9
    else:
        # How much data to use for training
        num_train = 50000

        # Optimization hyperparameters.
        batch_size = 100
        num_epochs = 4
        momentum = 0.9

    ##########################################################################
    # TODO: Set hyperparameters for training your model. You can change any  #
    # of the hyperparameters below.                                          #
    ##########################################################################
    lr = 0.035
    reg = 0.00000
    schedule = [15,22,25,28]
    lr_decay = 0.5
    
    lr = 0.030
    reg = 0.0005
    schedule = [1,2]
    lr_decay = 0.5
    
    ##########################################################################
    #                            END OF YOUR CODE                            #
    ##########################################################################

    train(args=args, num_train=num_train, hidden_dim=hidden_dim,
          weight_scale=weight_scale,
          batch_size=batch_size, num_epochs=num_epochs, momentum=momentum,
          lr=lr, reg=reg, schedule=schedule, lr_decay=lr_decay)

def train(args, num_train, hidden_dim, weight_scale, batch_size, num_epochs,
          momentum, lr, reg, schedule, lr_decay):

    data = load_fashion_mnist(num_train=num_train, seed=0)
    train_sampler = DataSampler(data['X_train'], data['y_train'], batch_size)
    val_sampler = DataSampler(data['X_val'], data['y_val'], batch_size)

    # Set up the model and optimizer
    np.random.seed(0)
    model = TwoLayerNet(hidden_dim=hidden_dim, weight_scale=weight_scale,
                        reg=reg)
    optimizer = SGD(model.parameters(), lr=lr, momentum=momentum)

    stats = {
        't': [],
        'loss': [],
        'train_acc': [],
        'val_acc': [],
    }

    for epoch in range(1, num_epochs + 1):
        print(f'Starting epoch {epoch} / {num_epochs}; lr: {optimizer.lr}')
        for i, (X_batch, y_batch) in enumerate(train_sampler):
            loss, grads = model.loss(X_batch, y_batch)
            optimizer.step(grads)
            if i == 0 or (i+1) % args.print_every == 0:
                print(f'  Iter {i} / {len(train_sampler)}, loss = {loss:.4f}')
            stats['t'].append(i / len(train_sampler) + epoch - 1)
            stats['loss'].append(loss)

        print('Checking accuracy')
        train_acc = check_accuracy(model, train_sampler)
        print(f'  Train: {train_acc:.2f}')
        val_acc = check_accuracy(model, val_sampler)
        print(f'  Val:   {val_acc:.2f}')
        stats['train_acc'].append(train_acc)
        stats['val_acc'].append(val_acc)
        
        if epoch in schedule:
            optimizer.lr *= lr_decay

    if args.action == 'overfit':
        plot_file = os.path.join('q3_result', args.plot_file + '_overfit.png')
        print(f'Saving plot to {plot_file}')
        plot_stats(stats, plot_file)
    else:
        plot_file = os.path.join('q3_result', args.plot_file + '.png')
        print(f'Saving plot to {plot_file}')
        plot_stats(stats, plot_file)
        checkpoint = os.path.join('q3_result', args.checkpoint + '.npy')
        print(f'Saving model checkpoint to {checkpoint}')
        model.save(checkpoint)


def check_accuracy(model, sampler):
    num_correct, num_samples = 0, 0
    for X_batch, y_batch in sampler:
        y_pred = model.predict(X_batch)
        num_correct += (y_pred == y_batch).sum()
        num_samples += y_pred.shape[0]
    acc = 100 * num_correct / num_samples
    return acc


def plot_stats(stats, filename):
    plt.subplot(1, 2, 1)
    plt.plot(stats['t'], stats['loss'], 'o', alpha=0.5, ms=4)
    plt.title('Loss')
    plt.xlabel('Epoch')
    loss_xlim = plt.xlim()

    plt.subplot(1, 2, 2)
    epoch = np.arange(1, 1 + len(stats['train_acc']))
    plt.plot(epoch, stats['train_acc'], '-o', label='train')
    plt.plot(epoch, stats['val_acc'], '-o', label='val')
    plt.xlim(loss_xlim)
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(loc='upper left')

    plt.gcf().set_size_inches(12, 4)
    plt.savefig(filename, bbox_inches='tight')
    plt.clf()


if __name__ == '__main__':
    os.makedirs('q3_result', exist_ok=True)

    main(parser.parse_args())
