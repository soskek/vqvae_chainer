#!/usr/bin/env python
"""Chainer example: train a VAE on MNIST
"""
from __future__ import print_function

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import argparse
import os

import chainer
from chainer import training
from chainer.training import extensions
import numpy as np

import net


def main():
    parser = argparse.ArgumentParser(description='Chainer example: VAE')
    parser.add_argument('--initmodel', '-m', default='',
                        help='Initialize the model from given file')
    parser.add_argument('--resume', '-r', default='',
                        help='Resume the optimization from snapshot')
    parser.add_argument('--gpu', '-g', default=-1, type=int,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--out', '-o', default='result',
                        help='Directory to output the result')
    parser.add_argument('--epoch', '-e', default=100, type=int,
                        help='number of epochs to learn')
    parser.add_argument('--dim-hidden', '-u', default=500, type=int,
                        help='dimention of hidden layers')
    parser.add_argument('--dimz', '-z', default=20, type=int,
                        help='dimention of encoded vector')
    parser.add_argument('--batchsize', '-b', type=int, default=100,
                        help='learning minibatch size')
    parser.add_argument('--test', action='store_true',
                        help='Use tiny datasets for quick tests')
    parser.add_argument('--vqvae', action='store_true',
                        help='Use VQVAE')
    args = parser.parse_args()

    print('GPU: {}'.format(args.gpu))
    print('# dim z: {}'.format(args.dimz))
    print('# Minibatch-size: {}'.format(args.batchsize))
    print('# epoch: {}'.format(args.epoch))
    print('')

    # Prepare VAE model, defined in net.py
    if args.vqvae:
        model = net.VQVAE(784, args.dimz, args.dim_hidden)
    else:
        model = net.VAE(784, args.dimz, args.dim_hidden)
    if args.gpu >= 0:
        chainer.cuda.get_device_from_id(args.gpu).use()
        model.to_gpu()

    # Setup an optimizer
    optimizer = chainer.optimizers.Adam(1e-4)
    optimizer.setup(model)
    optimizer.add_hook(chainer.optimizer.GradientClipping(5.))

    # Initialize
    if args.initmodel:
        chainer.serializers.load_npz(args.initmodel, model)

    # Load the MNIST dataset
    train, test = chainer.datasets.get_mnist(withlabel=False)
    if args.test:
        train, _ = chainer.datasets.split_dataset(train, 100)
        test, _ = chainer.datasets.split_dataset(test, 100)

    train_iter = chainer.iterators.SerialIterator(train, args.batchsize)
    test_iter = chainer.iterators.SerialIterator(test, args.batchsize,
                                                 repeat=False, shuffle=False)

    # Set up an updater. StandardUpdater can explicitly specify a loss function
    # used in the training with 'loss_func' option
    updater = training.StandardUpdater(train_iter, optimizer, device=args.gpu,
                                       loss_func=model.get_loss_func())

    trainer = training.Trainer(updater, (args.epoch, 'epoch'), out=args.out)
    trainer.extend(extensions.Evaluator(test_iter, model, device=args.gpu,
                                        eval_func=model.get_loss_func(k=10)))
    trainer.extend(extensions.dump_graph('main/loss'))
    trainer.extend(extensions.snapshot(), trigger=(args.epoch, 'epoch'))
    trainer.extend(extensions.LogReport())
    trainer.extend(extensions.PrintReport(
        ['epoch', 'main/loss', 'validation/main/loss',
         'main/rec_loss', 'validation/main/rec_loss',
         'main/other_loss', 'validation/main/other_loss',
         'elapsed_time']))
    trainer.extend(extensions.ProgressBar())

    if args.resume:
        chainer.serializers.load_npz(args.resume, trainer)

    @chainer.training.make_extension()
    def confirm_images(trainer):
        # Visualize the results
        def save_images(x, filename):
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(3, 3, figsize=(9, 9), dpi=100)
            for ai, xi in zip(ax.flatten(), x):
                ai.imshow(xi.reshape(28, 28))
            fig.savefig(filename)
            plt.close()

        model.to_cpu()
        train_ind = [1, 3, 5, 10, 2, 0, 13, 15, 17]
        x = chainer.Variable(np.asarray(train[train_ind]))
        with chainer.using_config('train', False), chainer.no_backprop_mode():
            x1 = model(x)
        save_images(x.data, os.path.join(
            args.out, '{.updater.iteration}_train'.format(trainer)))
        save_images(x1.data, os.path.join(
            args.out, '{.updater.iteration}_train_reconstructed'.format(trainer)))

        test_ind = [3, 2, 1, 18, 4, 8, 11, 17, 61]
        x = chainer.Variable(np.asarray(test[test_ind]))
        with chainer.using_config('train', False), chainer.no_backprop_mode():
            x1 = model(x)
        save_images(x.data, os.path.join(
            args.out, '{.updater.iteration}_test'.format(trainer)))
        save_images(x1.data, os.path.join(
            args.out, '{.updater.iteration}_test_reconstructed'.format(trainer)))

        # draw images from randomly sampled z
        if args.vqvae:
            z = model.sample(size=9)
        else:
            z = chainer.Variable(
                np.random.normal(0, 1, (9, args.dimz)).astype(np.float32))
        x = model.decode(z)
        save_images(x.data, os.path.join(
            args.out, '{.updater.iteration}_sampled'.format(trainer)))

        if args.gpu >= 0:
            chainer.cuda.get_device_from_id(args.gpu).use()
            model.to_gpu()

    trainer.extend(confirm_images, trigger=(args.epoch // 10, 'epoch'))

    # Run the training
    trainer.run()


if __name__ == '__main__':
    main()
