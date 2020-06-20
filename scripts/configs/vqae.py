# Similar as above but with max-pooling
def vqae():
    autoencoder = 'quantized'
    # Embedding size X embedding dimension
    latent_size = 256, 128
    beta = 1.0

    input_size = 1, 28, 28

    # n-channels, size, stride, padding
    encoder_layers = [
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

        ('flatten', [1]),

        ('linear', [256, False]),
        ('relu', []),
        ('batch_norm', [1]),

        ('linear', [256, False]),
        ('relu', []),
        ('batch_norm', [1])
    ]