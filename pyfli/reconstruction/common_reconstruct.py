import numpy as np


def mono_reconstruction(t, tau, S=1.0):
    return (S / tau) * np.exp(-t / tau)


def bi_reconstruction(t, tau1, tau2, A1, A2):
    return (A1 / tau1) * np.exp(-t / tau1) + (A2 / tau2) * np.exp(-t / tau2)


def mono_reconstruction_torch(t, tau, S=1.0):
    return (S / tau) * (-t / tau).exp()


def bi_reconstruction_torch(t, tau1, tau2, A1, A2):
    return (A1 / tau1) * (-t / tau1).exp() + (A2 / tau2) * (-t / tau2).exp()
