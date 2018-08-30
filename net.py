import six
import numpy as np

import chainer
import chainer.functions as F
from chainer.functions.loss.vae import gaussian_kl_divergence
import chainer.links as L


class VQVAE(chainer.Chain):
    """VQ Variational AutoEncoder"""

    def __init__(self, n_in, n_latent, n_h):
        super(VQVAE, self).__init__()

        n_vocab = 10  # K
        n_embed = 8  # the number of latent variables in different spaces

        with self.init_scope():
            # encoder
            self.le1 = L.Linear(n_in, n_h)
            self.le2 = L.Linear(n_h, n_h)
            self.le3 = L.Linear(n_h, n_latent)
            # decoder
            self.ld1 = L.Linear(n_latent, n_h)
            self.ld2 = L.Linear(n_h, n_h)
            self.ld3 = L.Linear(n_h, n_in)

            self.bne = L.BatchNormalization(
                n_latent, use_gamma=False, use_beta=True)
            # initialW=chainer.initializers.Normal()
            assert n_latent % n_embed == 0
            self.concat_embed = L.EmbedID(
                n_vocab * n_embed, n_latent // n_embed)

        self.n_latent = n_latent
        self.n_vocab = n_vocab
        self.n_embed = n_embed

    def __call__(self, x, sigmoid=True):
        """AutoEncoder"""
        return self.decode(self.encode(x), sigmoid)

    def encode(self, x):
        h = F.leaky_relu(self.le1(x))
        h = F.leaky_relu(self.le2(h))
        ze = self.le3(h)
        ze = self.bne(ze)

        def quantize_and_embed(ze, embed):
            batchsize, dim = ze.shape
            subdim = dim // self.n_embed
            ZE = ze.data[:, None, :]
            E = embed.W.data[None, :, :]

            # split dim to (subdim, n_embed)
            ZE = ZE.reshape(batchsize, 1, subdim, self.n_embed)
            E = self.xp.stack(self.xp.split(E, self.n_embed, axis=1), axis=-1)
            assert E.shape == (1, self.n_vocab, subdim, self.n_embed)

            argmin = self.xp.argmin(((ZE - E) ** 2).sum(axis=2), axis=1)
            assert argmin.shape == (batchsize, self.n_embed)
            argmin[:, self.xp.arange(self.n_embed)] \
                += self.xp.arange(self.n_embed) * self.n_vocab

            e = embed(argmin)
            ze = ze.reshape(batchsize, self.n_embed, subdim)
            # equivalent to F.stack(F.split_axis(ze, self.n_embed, axis=1), axis=1)
            loss = F.mean((ze.data - e) ** 2) + \
                F.mean((ze - e.data) ** 2) * 0.25
            loss *= e.shape[1]

            def ST(fw_val, bw_val):
                return fw_val.data + bw_val - bw_val.data
            e = ST(e, ze)
            return e, loss

        e, loss = quantize_and_embed(ze, self.concat_embed)
        self.other_loss = loss / x.shape[0]
        return e

    def decode(self, z, sigmoid=True):
        h = F.leaky_relu(self.ld1(z))
        h = F.leaky_relu(self.ld2(h))
        h = self.ld3(h)
        if sigmoid:
            return F.sigmoid(h)
        else:
            return h

    def sample(self, size):
        # TODO: get prior
        zis = self.xp.random.randint(
            0, self.n_vocab, (size, self.n_embed)).astype('i')
        zis[:, self.xp.arange(self.n_embed)] \
            += self.xp.arange(self.n_embed) * self.n_vocab

        e = self.concat_embed(zis)
        return e

    def get_loss_func(self, C=1.0, k=1):
        """Get loss function of VAE.

        The loss value is equal to ELBO (Evidence Lower Bound)
        multiplied by -1.

        Args:
            C (int): Usually this is 1.0. Can be changed to control the
                second term of ELBO bound, which works as regularization.
            k (int): Number of Monte Carlo samples used in encoded vector.
        """
        def lf(x):
            self.other_loss = 0.
            ze = self.encode(x)
            decoded_x = self.decode(ze, sigmoid=False)
            batchsize = x.shape[0]
            # self.rec_loss = F.mean_squared_error(x, decoded_x)
            self.rec_loss = F.bernoulli_nll(x, decoded_x) / batchsize
            self.loss = self.rec_loss + self.other_loss
            chainer.report(
                {'rec_loss': self.rec_loss,
                 'other_loss': self.other_loss,
                 'loss': self.loss}, observer=self)
            del self.rec_loss
            del self.other_loss
            return self.loss
        return lf


class VAE(chainer.Chain):
    """Variational AutoEncoder"""

    def __init__(self, n_in, n_latent, n_h):
        super(VAE, self).__init__()
        with self.init_scope():
            # encoder
            self.le1 = L.Linear(n_in, n_h)
            self.le2_mu = L.Linear(n_h, n_latent)
            self.le2_ln_var = L.Linear(n_h, n_latent)
            # decoder
            self.ld1 = L.Linear(n_latent, n_h)
            self.ld2 = L.Linear(n_h, n_in)

    def __call__(self, x, sigmoid=True):
        """AutoEncoder"""
        return self.decode(self.encode(x)[0], sigmoid)

    def encode(self, x):
        h1 = F.tanh(self.le1(x))
        mu = self.le2_mu(h1)
        ln_var = self.le2_ln_var(h1)  # log(sigma**2)
        return mu, ln_var

    def decode(self, z, sigmoid=True):
        h1 = F.tanh(self.ld1(z))
        h2 = self.ld2(h1)
        if sigmoid:
            return F.sigmoid(h2)
        else:
            return h2

    def get_loss_func(self, C=1.0, k=1):
        """Get loss function of VAE.

        The loss value is equal to ELBO (Evidence Lower Bound)
        multiplied by -1.

        Args:
            C (int): Usually this is 1.0. Can be changed to control the
                second term of ELBO bound, which works as regularization.
            k (int): Number of Monte Carlo samples used in encoded vector.
        """
        def lf(x):
            mu, ln_var = self.encode(x)
            batchsize = len(mu.data)
            # reconstruction loss
            rec_loss = 0
            for l in six.moves.range(k):
                z = F.gaussian(mu, ln_var)
                rec_loss += F.bernoulli_nll(x, self.decode(z, sigmoid=False)) \
                    / (k * batchsize)
            self.rec_loss = rec_loss
            self.loss = self.rec_loss + \
                C * gaussian_kl_divergence(mu, ln_var) / batchsize
            chainer.report(
                {'rec_loss': rec_loss, 'loss': self.loss}, observer=self)
            return self.loss
        return lf
