# VQVAE

[Neural Discrete Representation Learning](https://arxiv.org/pdf/1711.00937.pdf), Aaron van den Oord, Oriol Vinyals, Koray Kavukcuoglu, NIPS 2017

This is derived from [Chainer official VAE example](https://github.com/chainer/chainer/tree/master/examples/vae).


## MNIST reconstruction

Reconstuction from 2 embedded points, each of them are sampled from 50 discrete points. (50 patterns x 50 patterns)

```
python train_vae.py -g 0 --vqvae -z 20 -u 200
```

After training, you can see reconstructed digits as jpgs in directory `./result`.
