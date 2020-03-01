import numpy as np
import torch
import torchvision.transforms as trans
from torch.utils.data import DataLoader, random_split


def load_raw(path, latent_filter=None):
    data_zip = np.load(path, allow_pickle=True)

    imgs = data_zip['imgs']
    latents_values = data_zip['latents_values']
    latents_classes = data_zip['latents_classes']

    if latent_filter is not None:
        idx = latent_filter(latents_values)

        imgs = imgs[idx]
        latents_values = latents_classes[idx]
        latents_classes = latents_classes[idx]

    imgs = torch.from_numpy(imgs).to(dtype=torch.float32)
    latents_values = torch.from_numpy(latents_values).to(dtype=torch.float32)
    latents_classes = torch.from_numpy(latents_classes).to(dtype=torch.float32)

    return imgs, latents_classes, latents_values


class BatchGenerator:
    def __init__(self, data, batch_size, shuffle=True, random_state=None):
        if random_state is None or isinstance(random_state, int):
            random_state = np.random.RandomState(random_state)

        self.imgs, self.latent_values, self.classes = data
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.random_state = random_state

    def __len__(self):
        return len(self.imgs) // self.batch_size

    def __iter__(self):
        order = np.arange(len(self.imgs))
        if self.shuffle:
            self.random_state.shuffle(order)

        for i in range(len(self)):
            idx = np.arange(i * self.batch_size, (i + 1) * self.batch_size)
            yield self._get_batch(order[idx])

    def _get_batch(self, idx):
        imgs = self.imgs[idx]
        latent_values = self.latent_values[idx]
        latent_classes = self.classes[idx]

        return imgs, latent_values, latent_classes


class UnsupervisedLoader(BatchGenerator):
    def _get_batch(self, idx):
        imgs, latent_values, latent_classes = super()._get_batch(idx)
        imgs = imgs.reshape(-1, 64 * 64)
        return imgs, imgs


class SupervisedLoader(BatchGenerator):
    def _get_batch(self, idx):
        imgs, latent_values, latent_classes = super()._get_batch(idx)
        imgs = imgs.reshape(-1, 64 * 64)
        return imgs, latent_classes


class SemiSupervisedLoader(BatchGenerator):
    def _get_batch(self, idx):
        imgs, latent_values, latent_classes = super()._get_batch(idx)
        imgs = imgs.reshape(-1, 64 * 64)
        return imgs, (latent_classes, imgs)


class ValidationWrapper:
    def __init__(self, generator, n_batches):
        self.generator = generator
        self.n_batches = n_batches

    def __len__(self):
        return self.n_batches

    def __iter__(self):
        for i, batch in enumerate(self.generator):
            if i >= self.n_batches:
                break
            yield batch


def get_loader(setting):
    if setting == 'unsupervised':
        return UnsupervisedLoader
    elif setting == 'supervised':
        return SupervisedLoader
    elif setting == 'semi-supervised':
        return SemiSupervisedLoader
    raise ValueError('Unrecognized setting "{}"'.format(setting))


def load_dsprites(path, setting, batch_size=32, data_filters=(None, None), 
                  val_ratio=0.2, shuffle=True, random_state=None):
    train_filter, test_filter = data_filters

    train_data = load_raw(path, train_filter)
    test_data = train_data if test_filter is None else load_raw(path,test_filter)
    val_n_batches = np.ceil(len(train_data) * val_ratio / batch_size)

    loader = get_loader(setting)

    train_data = loader(train_data, batch_size, True, random_state)
    test_data = loader(test_data, batch_size, True, random_state)
    val_data = ValidationWrapper(train_data, val_n_batches)

    return train_data, test_data, val_data
