from PIL import Image
import torchvision.transforms as trans
import torchvision.datasets as dataset
from torch.utils.data import DataLoader, random_split


class UnsupervisedMNIST:
    def __init__(self, data, transform=None, target_transform=None):
        self.data = data
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is the same image
        """
        img = self.data[index]
        img = Image.fromarray(img.numpy(), mode='L')

        if self.transform is not None:
            img = self.transform(img)

        target = img

        return img, target


def load_raw(data_path, input_size, train=True, download=False, transform=None):
    transform = transform or trans.Compose([
        trans.ToTensor(),
        trans.Lambda(lambda x: x.view(input_size))
    ])

    data = dataset.MNIST(
        root=data_path, train=train, transform=transform, download=download)

    return data


def load_unsupervised(data_path, input_size, train=True, download=False, transform=None):
    data = load_raw(data_path, input_size, train, download, transform)

    data = UnsupervisedMNIST(data.data, data.transform)

    return data


def load_mnist(data_path, input_size, batch_size, val_split=0.2,
               shuffle=True, train=True, download=True, supervised=True
    ):

    if supervised:
        raw = load_raw(data_path, input_size, train, download)
    else:
        raw = load_unsupervised(data_path, input_size,train, download)

    if train:
        # Split train data into training and validation sets
        N = len(raw)
        val_size = int(N * val_split)
        train_raw, validation_raw = random_split(
            raw, [N - val_size, val_size])

        train_data = DataLoader(
            train_raw, batch_size=batch_size, shuffle=shuffle)
        validation_data = DataLoader(
            validation_raw, batch_size=batch_size, shuffle=False)


        return train_data, validation_data

    test_data = DataLoader(raw, batch_size=batch_size, shuffle=False)

    return test_data
