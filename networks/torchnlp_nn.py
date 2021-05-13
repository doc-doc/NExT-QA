from torch.nn import Parameter

import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class weight_drop():

    def __init__(self, module, weights, dropout):
        for name_w in weights:
            w = getattr(module, name_w)
            del module._parameters[name_w]
            module.register_parameter(name_w + '_raw', Parameter(w))

        self.original_module_forward = module.forward

        self.weights = weights
        self.module = module
        self.dropout = dropout

    def __call__(self, *args, **kwargs):
        for name_w in self.weights:
            raw_w = getattr(self.module, name_w + '_raw')
            w = torch.nn.functional.dropout(
                raw_w, p=self.dropout, training=self.module.training)
            # module.register_parameter(name_w, Parameter(w))
            setattr(self.module, name_w, Parameter(w))

        return self.original_module_forward(*args, **kwargs)


def _weight_drop(module, weights, dropout):
    setattr(module, 'forward', weight_drop(module, weights, dropout))


# def _weight_drop(module, weights, dropout):
#     """
#     Helper for `WeightDrop`.
#     """

#     for name_w in weights:
#         w = getattr(module, name_w)
#         del module._parameters[name_w]
#         module.register_parameter(name_w + '_raw', Parameter(w))

#     original_module_forward = module.forward

#     def forward(*args, **kwargs):
#         for name_w in weights:
#             raw_w = getattr(module, name_w + '_raw')
#             w = torch.nn.functional.dropout(
#                 raw_w, p=dropout, training=module.training)
#             # module.register_parameter(name_w, Parameter(w))
#             setattr(module, name_w, Parameter(w))

#         return original_module_forward(*args, **kwargs)

#     setattr(module, 'forward', forward)


class WeightDrop(torch.nn.Module):
    """
    The weight-dropped module applies recurrent regularization through a DropConnect mask on the
    hidden-to-hidden recurrent weights.
    **Thank you** to Sales Force for their initial implementation of :class:`WeightDrop`. Here is
    their `License
    <https://github.com/salesforce/awd-lstm-lm/blob/master/LICENSE>`__.
    Args:
        module (:class:`torch.nn.Module`): Containing module.
        weights (:class:`list` of :class:`str`): Names of the module weight parameters to apply a
          dropout too.
        dropout (float): The probability a weight will be dropped.
    Example:
        >>> from torchnlp.nn import WeightDrop
        >>> import torch
        >>>
        >>> torch.manual_seed(123)
        <torch._C.Generator object ...
        >>>
        >>> gru = torch.nn.GRUCell(2, 2)
        >>> weights = ['weight_hh']
        >>> weight_drop_gru = WeightDrop(gru, weights, dropout=0.9)
        >>>
        >>> input_ = torch.randn(3, 2)
        >>> hidden_state = torch.randn(3, 2)
        >>> weight_drop_gru(input_, hidden_state)
        tensor(... grad_fn=<AddBackward0>)
    """

    def __init__(self, module, weights, dropout=0.0):
        super(WeightDrop, self).__init__()
        _weight_drop(module, weights, dropout)
        self.forward = module.forward


class WeightDropLSTM(torch.nn.LSTM):
    """
    Wrapper around :class:`torch.nn.LSTM` that adds ``weight_dropout`` named argument.
    Args:
        weight_dropout (float): The probability a weight will be dropped.
    """

    def __init__(self, *args, weight_dropout=0.0, **kwargs):
        super().__init__(*args, **kwargs)
        weights = ['weight_hh_l' + str(i) for i in range(self.num_layers)]
        _weight_drop(self, weights, weight_dropout)


class WeightDropGRU(torch.nn.GRU):
    """
    Wrapper around :class:`torch.nn.GRU` that adds ``weight_dropout`` named argument.
    Args:
        weight_dropout (float): The probability a weight will be dropped.
    """

    def __init__(self, *args, weight_dropout=0.0, **kwargs):
        super().__init__(*args, **kwargs)
        weights = ['weight_hh_l' + str(i) for i in range(self.num_layers)]
        _weight_drop(self, weights, weight_dropout)


class WeightDropLinear(torch.nn.Linear):
    """
    Wrapper around :class:`torch.nn.Linear` that adds ``weight_dropout`` named argument.
    Args:
        weight_dropout (float): The probability a weight will be dropped.
    """

    def __init__(self, *args, weight_dropout=0.0, **kwargs):
        super().__init__(*args, **kwargs)
        weights = ['weight']
        _weight_drop(self, weights, weight_dropout)
