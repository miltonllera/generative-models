# Similar as above but with max-pooling
def cnn_vae():
    input_size = 1, 28, 28

    cnn_layers = [
        # n-channels, size, stride, padding
        ('conv', (32, 3, 1, 1)),
        # size, stride, padding, type
        ('pool', (2, 2, 0, 'max')),
        ('relu', []),
        ('conv', (32, 3, 1, 1)),
        ('pool', (2, 2, 0, 'max')),
        ('relu', []),
        ('conv', (32, 3, 1, 1)),
        ('pool', (2, 2, 0, 'max')),
        ('relu', []),
        ('conv', (32, 3, 1, 1)),
        ('pool', (2, 2, 0, 'max')),
        ('relu', []),
    ]

    encoder_sizes = 256, 256
    latent_size = 10
    autoencoder = 'variational'
    batch_norm = False

    init_fn = 'cnn-vae'