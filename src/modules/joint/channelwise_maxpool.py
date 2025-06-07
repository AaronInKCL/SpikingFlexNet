import torch
from torch import cat


def channel_wise_maxpool(tensor_1, tensor_2):
    """
    Take two tensors of identical shape and return a tensor of the same shape using element-wise max pooling.
    Also returns the ratio of values from tensor_2 to the total.
    """
    assert tensor_1.shape == tensor_2.shape, "tensor_1 and tensor_2 must have the same shape"

    joint = cat([tensor_1.unsqueeze(-1), tensor_2.unsqueeze(-1)], dim=-1)
    pooled, indices = torch.max(joint, dim=-1)

    # count values are from conv (tensor 2)
    with torch.no_grad():
        conv_ratio = (indices == 1).sum().item() / tensor_1.numel()
        cp_identity_matrix = (indices == 1).int()

    return pooled, conv_ratio, cp_identity_matrix


if __name__ == "__main__":
    tensor1 = torch.rand(8, 3, 32, 32)
    tensor2 = torch.rand(8, 3, 32, 32)
    joint, ratio, cp_identity_matrix = channel_wise_maxpool(tensor1, tensor2)

    print(joint.shape)
    print(f"Ratio of values from tensor1 to total: {ratio:.4f}")
