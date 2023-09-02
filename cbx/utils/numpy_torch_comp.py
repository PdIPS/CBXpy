def copy_particles(x, mode='numpy'):
    r"""Copy particles
    
    Parameters
    ----------
    x : numpy.ndarray
        Array of particles of shape (N, d)
    mode : str, optional
        Method for copying the particles. The default is "numpy".
        Possible values: "numpy", "torch"
    
    Returns
    -------
    x : numpy.ndarray
        Array of particles of shape (N, d)
    """
    if mode == 'numpy':
        return x.copy()
    elif mode == 'torch':
        return x.clone()
    else:
        raise Exception('Unknown mode for copy_particles specified!')