import torch
import torch.nn as nn


class Biaffine(nn.Module):
    r"""
  Biaffine layer for first-order scoring.

  This function has a tensor of weights :math:`W` and bias terms if needed.
  The score :math:`s(x, y)` of the vector pair :math:`(x, y)` is computed as :math:`x^T W y`,
  in which :math:`x` and :math:`y` can be concatenated with bias terms.

  References:
      - Timothy Dozat and Christopher D. Manning. 2017.
        `Deep Biaffine Attention for Neural Dependency Parsing`_.

  Args:
      x_features (int):
          size of each first input sample.
      y_features (int):
          size of each second input sample.
      out_features (int):
          size of each output sample.
      bias_x (bool):
          If ``True``, adds a bias term for tensor :math:`x`. Default: ``True``.
      bias_y (bool):
          If ``True``, adds a bias term for tensor :math:`y`. Default: ``True``.

  .. _Deep Biaffine Attention for Neural Dependency Parsing:
      https://openreview.net/forum?id=Hk95PK9le
  """

    def __init__(self, x_features, y_features, out_features=1, bias_x=True, bias_y=True):
        super().__init__()

        self.x_features = x_features
        self.y_features = y_features
        self.out_features = out_features
        self.bias_x = bias_x
        self.bias_y = bias_y
        self.weight = nn.Parameter(torch.Tensor(out_features, x_features + bias_x, y_features + bias_y))

        self.reset_parameters()

    def __repr__(self):
        s = f"x_features={self.x_features}, y_features={self.y_features}, out_features={self.out_features}"
        if self.bias_x:
            s += f", bias_x={self.bias_x}"
        if self.bias_y:
            s += f", bias_y={self.bias_y}"

        return f"{self.__class__.__name__}({s})"

    def reset_parameters(self):
        nn.init.zeros_(self.weight)

    def forward(self, x, y):
        r"""
    Args:
        x (torch.Tensor): ``[batch_size, seq_len, x_features]``.
        y (torch.Tensor): ``[batch_size, seq_len, y_features]``.

    Returns:
        ~torch.Tensor:
            A scoring tensor of shape ``[batch_size, out_features, seq_len, seq_len]``.
            If ``out_features=1``, the dimension for ``out_features`` will be squeezed automatically.
    """

        if self.bias_x:
            x = torch.cat((x, torch.ones_like(x[..., :1])), -1)
        if self.bias_y:
            y = torch.cat((y, torch.ones_like(y[..., :1])), -1)
        # [batch_size, out_features, seq_len, seq_len]
        s = torch.einsum('bxi,oij,byj->boxy', x, self.weight, y)
        # remove dim 1 if out_features == 1
        s = s.squeeze(1)

        return s


def sequence_mask(lengths, max_seq_length=None, dtype=torch.bool):
    if max_seq_length is None:
        max_seq_length = lengths.max()
    row_vector = torch.arange(0, max_seq_length, 1)
    matrix = torch.unsqueeze(lengths, dim=-1)
    mask = row_vector < matrix

    mask.type(dtype)
    return mask
