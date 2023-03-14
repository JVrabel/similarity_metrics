from jax import jit
import numpy as np

@jit
def mutual_info(x, y):
    hgram, _, _ = jnp.histogram2d(x, y, bins=100)
    hgram += 1e-8
    pxy = hgram / jnp.sum(hgram)
    px = jnp.sum(pxy, axis=1)
    py = jnp.sum(pxy, axis=0)
    px_py = px[:, None] * py[None, :]
    return jnp.sum(pxy* jnp.log(pxy / px_py))


@jit
def kl_div_symm(x, y, nbins=1000):
    # Compute histograms for each input
    hist_x, _ = jnp.histogram(x, bins=nbins, range = (x.max()*0.01, x.max()*0.99))#, density=True)
    hist_y, _ = jnp.histogram(y, bins=nbins, range = (y.max()*0.01, y.max()*0.99))#, density=True)

    # Add small constant value to each bin count to avoid zero bins
    hist_x = jnp.maximum(hist_x, 1e-8)
    hist_y = jnp.maximum(hist_y, 1e-8)

    # Compute KL divergences
    kl_xy = jnp.sum(hist_x * jnp.log(hist_x / hist_y))
    kl_yx = jnp.sum(hist_y * jnp.log(hist_y / hist_x))

    # Compute symmetric KL divergence similarity score
    score = (kl_xy + kl_yx) / 2.0
    
    return score

@jit
def phase_corr(x, y):
    """Compute the normalized phase correlation between two signals."""

    # Compute the Fourier transform of the images
    f1 = jnp.fft.fft(x)
    f2 = jnp.fft.fft(y)

    # normalize the Fourier transforms
    f1_norm = f1 / jnp.sqrt(jnp.sum(jnp.abs(f1)**2))
    f2_norm = f2 / jnp.sqrt(jnp.sum(jnp.abs(f2)**2))

    # Compute the cross-power spectrum
    cross_power_spectrum = f1_norm * jnp.conj(f2_norm)

    # Compute the magnitude of the cross-power spectrum
    magnitude = jnp.abs(cross_power_spectrum)

    # Normalize the cross-power spectrum by the magnitude
    norm_cross_power_spectrum = cross_power_spectrum# / (magnitude + 1e-8)

    # Compute the inverse Fourier transform to get the normalized phase correlation
    norm_xcorr = jnp.fft.ifft(norm_cross_power_spectrum)

    # Shift the output to center the peak
    norm_xcorr = jnp.fft.fftshift(norm_xcorr)

    return 1 - jnp.sum(jnp.real(norm_xcorr))



def cosine(a, b):
    return 1 - jnp.dot(a, b) / jnp.linalg.norm(a) / jnp.linalg.norm(b)

# def euclid(a, b):
#     return jnp.linalg.norm(a - b)
def euclid(a, b):
    return np.linalg.norm(a - b)

def norm_mutual_info(x, y):
    return mutual_info(x,y)/((1/2)*(mutual_info(x,x)+mutual_info(y,y)))


def gram_rectangular(kernel_func, X, Y):
  mv = vmap(kernel_func, (0, None), 0) #  ([b,a], [a]) -> [b]      (b is the mapped axis)
  mm = vmap(mv, (None, 1), 1)(X,Y) #  ([b,a], [a,c]) -> [b,c]  (c is the mapped axis)
  return mm.T


