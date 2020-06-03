import sys
import matplotlib.pyplot as plt
from sacred import Ingredient

if sys.path[0] != '../src':
    sys.path.insert(0, '../src')

from dataset.mnist import load_mnist

dataset = Ingredient('dataset')
load_mnist = dataset.capture(load_mnist)

dataset.add_config(setting='unsupervised')
dataset.add_named_config('unsupervised', setting='unsupervised')
dataset.add_named_config('supervised', setting='supervised')


@dataset.capture
def get_dataset_loader(dataset_name):
    if dataset_name == 'mnist':
        dataset_loader = load_mnist
    else:
        raise ValueError('Unrecognized dataset {}'.format(dataset_name))

    return dataset_loader


@dataset.command(unobserved=True)
def plot():
    dataset = get_dataset_loader()(condition='train', setting='supervised',
                                   batch_size=1)[0]

    for img, t in dataset:
        img = img.numpy()

        if len(img.shape) == 3:
            img.transpose(2, 1, 0).squeeze()
            cmap=None
        else:
            cmap='Greys_r'

        plt.imshow(img.reshape(64, 64),cmap=cmap, vmin=0, vmax=1)
        plt.show()
