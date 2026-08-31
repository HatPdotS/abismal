import tensorflow as tf
from tensorflow_probability import util as tfu
from tensorflow_probability import bijectors as tfb
import tf_keras as tfk

def weighted_pearsonr(x, y, w=None, axis=-1, keepdims=False, eps=1e-12):
    """
    Calculate a [weighted Pearson correlation coefficient](https://en.wikipedia.org/wiki/Pearson_correlation_coefficient#Weighted_correlation_coefficient).

    Note
    ----
    x, y, and w may have arbitrarily shaped leading dimensions. The correlation coefficient will always be computed pairwise along the last axis.

    Parameters
    ----------
    x : np.array(float)
        An array of observations.
    y : np.array(float)
        An array of observations the same shape as x.
    w : np.array(float)
        An array of weights the same shape as x. These needn't be normalized.

    Returns
    -------
    r : float
        The Pearson correlation coefficient along the last dimension. This has shape {x,y,w}.shape[:-1].
    """
    if w is None:
        w = tf.ones_like(x)

    z = tf.math.reciprocal(tf.reduce_sum(w, axis=axis, keepdims=True))
    mx = tf.reduce_sum(z * (w * x), axis=axis, keepdims=True)
    my = tf.reduce_sum(z * (w * y), axis=axis, keepdims=True)

    dx = x - mx
    dy = y - my

    cxy = z * tf.reduce_sum(w * dx * dy, axis=axis, keepdims=True)
    cx = z * tf.reduce_sum(w * dx * dx, axis=axis, keepdims=True)
    cy = z * tf.reduce_sum(w * dy * dy, axis=axis, keepdims=True)

    r = cxy / tf.sqrt(cx * cy + eps)
    return r

class LocationScale(tfk.layers.Layer):
    def _likelihood(self, iobs, sigiobs):
        raise NotImplementedError(
            "Derived classes must implement _likelihood(iobs, sigiobs)->tfd.Distribution"
        )

    def register_metrics(self, ipred, iobs, sigiobs):
        #likelihood = self._likelihood(iobs, sigiobs)

        # This is the mean ipred across the posterior mc samples
        iobs = iobs * tf.ones_like(ipred)
        sigiobs = sigiobs * tf.ones_like(ipred)
        w = tf.math.reciprocal(tf.square(sigiobs))

        cc = weighted_pearsonr(iobs, ipred, w, axis=(-2, -1))
        cc = tf.squeeze(cc)
        self.add_metric(cc, name='wCCpred')

        cc = weighted_pearsonr(iobs, ipred, axis=(-2, -1))
        cc = tf.squeeze(cc)
        self.add_metric(cc, name='CCpred')


        resid = ipred - iobs
        r2 = tf.square(resid)

        #mse = tf.reduce_mean(r2)
        #self.add_metric(mse, name='MSE')

        #mae = tf.reduce_mean(tf.abs(resid))
        #self.add_metric(mse, name='MAE')

        #wmse = tf.reduce_sum(w * r2) / tf.reduce_sum(w * tf.ones_like(r2))
        #self.add_metric(wmse, name='WMSE')


    def call(self, ipred, iobs, sigiobs):
        likelihood = self._likelihood(iobs, sigiobs)
        ll = likelihood.log_prob(ipred)
        return ll

class Ev11Likelihood(LocationScale):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Ev11: sig'^2 = Sdfac^2 * (sig^2 + SdB*ipred + Sdadd*ipred^2)
        # Sdadd is a VARIANCE coefficient, so it is the square of the fractional
        # error. Upstream initialises all three at 1.0, which makes sig' ~ ipred,
        # caps I/sigma at ~1 and destroys the signal before refinement can relax
        # it. Our data says the useful fractional error is 0.1-0.2 (it takes the
        # effective number of reflections determining a crystal's scale from 4.3
        # to 8-12), so Sdadd = 0.2^2 = 0.04. SdB starts near zero because the
        # Poisson term is already inside CrystFEL's sigma.
        self.Sdfac = tfu.TransformedVariable(1.,   tfb.Softplus())
        self.Sdadd = tfu.TransformedVariable(0.04, tfb.Softplus())
        self.SdB   = tfu.TransformedVariable(0.01, tfb.Softplus())
        self.built = True

    def corrected_sigiobs(self, ipred, sigiobs):
        ipred = tf.stop_gradient(ipred)
        ipred = tf.maximum(ipred, 0.)
        sigiobs = self.Sdfac * tf.math.sqrt(
            tf.square(sigiobs) + \
            self.SdB * ipred + \
            self.Sdadd * tf.square(ipred)
        )
        return sigiobs

    def call(self, ipred, iobs, sigiobs):
        corrected_sigiobs = self.corrected_sigiobs(ipred, sigiobs)
        likelihood = self._likelihood(iobs, corrected_sigiobs)
        ll = likelihood.log_prob(ipred)
        return ll
