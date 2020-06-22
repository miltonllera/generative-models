# Similar as above but with max-pooling
def vqae():
    autoencoder = 'quantized'
    # Embedding size X embedding dimension
    latent_size = 256, 128
    input_size = 1, 28, 28

    # n-channels, size, stride, padding
    encoder_layers = [
        # n-channels, size, stride, padding
        ('conv', (32, 3, 2, 1)),
        # size, stride, padding, type
        ('relu', []),
        ('conv', (32, 3, 2, 1)),
        ('relu', []),
        ('conv', (32, 3, 2, 1)),
        ('relu', []),

        ('flatten', [1]),

        ('linear', [256]),
        ('relu', []),
        ('linear', [256]),
        ('relu', []),

        ('linear', [256])
    ]

def pi_vqae():
    autoencoder = 'quantized'
    # Embedding size X embedding dimension
    latent_size = 64, 128
    input_size = 1, 28, 28

    # n-channels, size, stride, padding
    encoder_layers = [
        # n-channels, size, stride, padding
        ('conv', (32, 3, 1, 0)),
        ('relu', []),
        # size, stride, padding, type
        ('conv', (64, 3, 1, 0)),
        ('relu', []),
        ('conv', (64, 3, 1, 0)),
        ('relu', []),
        ('conv', (128, 3, 1, 0)),
        ('relu', []),
        ('transpose', [1, 3]),
        ('flatten', [1, 2]),
]