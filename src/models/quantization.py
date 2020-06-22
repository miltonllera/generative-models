import torch
import torch.nn as nn


def distmin(inputs, distance, idx):
    dist = dist[:, idx]
    # dist.sum().backward()?
    dist.backward(gradient=torch.ones_like(dist))


class EMAUpdater(nn.Module):
    # __slots__ = '_N', '_m', 'codebook', 'gamma'

    def __init__(self, codebook, gamma=0.99):
        super().__init__()

        book_size, code_size = codebook.size()
        N = torch.ones((book_size, 1), dtype=torch.float32)
        m = torch.zeros((book_size, code_size), dtype=torch.float32)

        self.register_buffer('_N', N)
        self.register_buffer('_m', m)

        self.codebook = codebook
        # self.codebook.requires_grad_(False)
        self.gamma = gamma

    def __call__(self, inputs, distances, idx):
        # Compute counts
        embedding_idx, counts = torch.unique(idx, return_counts=True)
        new_counts = torch.zeros_like(self._N)

        counts = counts.to(dtype=torch.float32).unsqueeze_(1)

        new_counts[embedding_idx] = counts

        # Ceate a mask of size (n_embeddings, batch_size) where for each each
        # row `i` the columns are 1 if the corresponding input are assigned to
        # embedding `i`. A matrix multiplication produces the aggregates.
        batch_size = inputs.size(0)
        mask = inputs.new_zeros((self.codebook.size(0), batch_size))
        mask[idx, torch.arange(batch_size)] = 1.0

        new_m = mask.matmul(inputs)

        # Update moving averages
        self._N = self.gamma * self._N + (1 - self.gamma) * new_counts
        self._m = self.gamma * self._m + (1 - self.gamma) * new_m

        # Update the embedding
        self.codebook.data = (self._m / self._N).data

    def reset_parameter(self):
        self._N.fill_(1.0)
        self._m.zero_()


class Quantization(nn.Module):
    def __init__(self, book_size, code_size, beta=1.0, update_type='expmavg'):
        if update_type not in ['expmavg', 'distmin']:
            raise ValueError('Unrecognized update {}'.format(update))

        super().__init__()

        self.beta = beta
        self.update_type = update_type
        self.codebook = nn.Parameter(torch.empty((book_size, code_size),
                                                 dtype=torch.float32))

        if update_type == 'expmavg':
            self.update_code = EMAUpdater(self.codebook)
        elif update_type == 'euclid':
            self.update_code = distmin
        else:
            self.update_code = update_type
            self.update_type = str(update_type)

        self.reset_parameters()

    @property
    def book_size(self):
        return self.codebook.size(0)

    @property
    def code_size(self):
        return self.codebook.size(1)

    def reset_parameters(self):
        lim = 1 / self.book_size
        nn.init.uniform_(self.codebook, -lim, lim)
        if isinstance(self.update_code, nn.Module):
            self.update_code.reset_parameter()

    def forward(self, inputs):
        input_size = inputs.size()

        if len(input_size) > 2:
            inputs = inputs.flatten(0, 1).contiguous()

        detached_input = inputs.detach()
        embeddings = self.codebook.unsqueeze(0)

        # Input to cdist is BxPxM, BxRxM so we unsqueeze dim 0
        # So we have inputs 1xNxD, 1xBxD and output 1xNxB
        # print(detached_input.size())
        # print(embeddings.size())
        dist = torch.cdist(detached_input.unsqueeze(0), embeddings, p=2)

        idx = dist.squeeze_().argmax(dim=1)
        quantized = self.codebook[idx].contiguous() # z_q

        if self.training:
            self.update_code(detached_input, dist, idx)
            quantized = inputs + (quantized - inputs).detach_()

            # Propagate the gradients. First minimize input-embedding distance
            # by treating the latter as parameters in the computation. Then,
            # backpropagate the gradient w.r.t. the decoder through the input.
            # def compute_commitment_loss(grad):
                # nonlocal inputs, detached_inputs, quantized, dist, idx
                # with torch.enable_grad():
                    # detached_input.requires_grad_(True)
                    # commit_loss = (detached_input - quantized.detach()).pow(2)
                    # commit_loss = self.beta * commit_loss.mean(0).sum()
                    # commit_loss.backward()

                # self.update_code(detached_input, dist, idx)

                # return grad + detached_input.grad

                # Commit loss = beta * 1/N * \sum_N (input - quantized)^2
                # Commit loss grad wrt input = beta * 2/N * (input - quantized)
                # size = inputs.size(0)
                # commit_loss_grad = 2 * self.beta * (inputs - quantized) / size

                # self.update_code(detached_input, dist, idx)

                # return grad + commit_loss_grad

            # quantized = inputs + (quantized - inputs).detach_()
            # inputs.register_hook(compute_commitment_loss)

        if len(input_size) > 2:
            quantized = quantized.reshape(input_size).contiguous()

        return quantized

    def extra_repr(self):
        return 'book size={}, code size={}, beta={}, code_update={}'.format(
            self.book_size, self.code_size, self.beta, self.update_type
        )
