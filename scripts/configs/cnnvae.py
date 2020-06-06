# Similar as above but with max-pooling
def cnn_vae():
    input_size = 1, 28, 28

    # n-channels, size, stride, padding
    kernels = [(32, 3, 1, 1),
               (32, 3, 1, 1),
               (32, 3, 1, 1),
               (32, 3, 1, 1)]

    # size, stride, padding, type
    pools = [(2, 2, 0, 'max'),
             (2, 2, 0, 'max'),
             (2, 2, 0, 'max'),
             (2, 2, 0, 'max')]

    encoder_sizes = 256, 256, 20
    latent_size = 10
    autoencoder = 'variational'
    batch_norm = False

    init_fn = 'cnn-vae'