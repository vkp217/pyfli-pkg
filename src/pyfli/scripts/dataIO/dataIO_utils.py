# dataIO_utils.py 
import h5py

class DataIO_utils:
    def __init__(self):
        pass

    def load_phasors_hdf5(self, file_path):
        with h5py.File(file_path, 'r') as f:
            Gc = f['Gc'][:]
            Sc = f['Sc'][:]
            tau = f['tau'][:] if 'tau' in f else None
        if tau is not None:
            if Gc.shape[1:] != tau.shape:
                raise ValueError(f"Dimension mismatch: Phasor spatial size {Gc.shape[1:]} "
                    f"does not match Tau size {tau.shape}.")
            if Gc.shape != Sc.shape:
                raise ValueError("Critical Error: Gc and Sc dimensions do not match.")
        return Gc, Sc, tau