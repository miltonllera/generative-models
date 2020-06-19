# Similar as above but with max-pooling
def vqae():
    input_size = 1, 28, 28

    # n-channels, size, stride, padding
    kernels = [(16, 3, 1, 1),
               (16, 3, 1, 1),
               (16, 3, 1, 1)]

    # size, stride, padding, type
    pools = [(2, 2, 1, 'max'),
             (2, 2, 0, 'max'),
             (2, 1, 0, 'max')]

    encoder_sizes = 256, 256, 128
    code_size = 128
    book_size = 256
    beta = 1.0
    autoencoder = 'quantized'
    batch_norm = False

    init_fn = 'cnn-vae'