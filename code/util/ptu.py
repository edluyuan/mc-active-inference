
import torch
import numpy as np
from torch.nn import functional as F
import railrl.misc.gpu_util as gpu_util
import os
import pdb
import logging

log = logging.getLogger(os.path.basename(__file__))

def soft_update_from_to(source, target, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(
            target_param.data * (1.0 - tau) + param.data * tau
        )

def copy_model_params_from_to(source, target):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)

def maximum_2d(t1, t2):
    # noinspection PyArgumentList
    return torch.max(
        torch.cat((t1.unsqueeze(2), t2.unsqueeze(2)), dim=2),
        dim=2,
    )[0].squeeze(2)

def kronecker_product(t1, t2):
    """
    Computes the Kronecker product between two tensors
    See https://en.wikipedia.org/wiki/Kronecker_product
    """
    t1_height, t1_width = t1.size()
    t2_height, t2_width = t2.size()
    out_height = t1_height * t2_height
    out_width = t1_width * t2_width

    tiled_t2 = t2.repeat(t1_height, t1_width)
    expanded_t1 = (
        t1.unsqueeze(2)
          .unsqueeze(3)
          .repeat(1, t2_height, t2_width, 1)
          .view(out_height, out_width)
    )

    return expanded_t1 * tiled_t2

def selu(
        x,
        alpha=1.6732632423543772848170429916717,
        scale=1.0507009873554804934193349852946,
):
    """
    Based on https://github.com/dannysdeng/selu/blob/master/selu.py
    """
    return scale * (
        F.relu(x) + alpha * (F.elu(-1 * F.relu(-1 * x)))
    )

def softplus(x):
    """
    PyTorch's softplus isn't (easily) serializable.
    """
    return F.softplus(x)

def alpha_dropout(
        x,
        p=0.05,
        alpha=-1.7580993408473766,
        fixedPointMean=0,
        fixedPointVar=1,
        training=False,
):
    keep_prob = 1 - p
    if keep_prob == 1 or not training:
        return x
    a = np.sqrt(fixedPointVar / (keep_prob * (
        (1 - keep_prob) * pow(alpha - fixedPointMean, 2) + fixedPointVar)))
    b = fixedPointMean - a * (
        keep_prob * fixedPointMean + (1 - keep_prob) * alpha)
    keep_prob = 1 - p

    random_tensor = keep_prob + torch.rand(x.size())
    binary_tensor = torch.floor(random_tensor)
    x = x.mul(binary_tensor)
    ret = x + alpha * (1 - binary_tensor)
    ret.mul_(a).add_(b)
    return ret

def alpha_selu(x, training=False):
    return alpha_dropout(selu(x), training=training)

def double_moments(x, y):
    """
    Returns the first two moments between x and y.

    Specifically, for each vector x_i and y_i in x and y, compute their
    outer-product. Flatten this resulting matrix and return it.

    The first moments (i.e. x_i and y_i) are included by appending a `1` to x_i
    and y_i before taking the outer product.
    :param x: Shape [batch_size, feature_x_dim]
    :param y: Shape [batch_size, feature_y_dim]
    :return: Shape [batch_size, (feature_x_dim + 1) * (feature_y_dim + 1)
    """
    batch_size, x_dim = x.size()
    _, y_dim = x.size()
    x = torch.cat((x, torch.ones(batch_size, 1)), dim=1)
    y = torch.cat((y, torch.ones(batch_size, 1)), dim=1)
    x_dim += 1
    y_dim += 1
    x = x.unsqueeze(2)
    y = y.unsqueeze(1)

    outer_prod = (
        x.expand(batch_size, x_dim, y_dim) * y.expand(batch_size, x_dim, y_dim)
    )
    return outer_prod.view(batch_size, -1)

def batch_diag(diag_values, diag_mask=None):
    batch_size, dim = diag_values.size()
    if diag_mask is None:
        diag_mask = torch.diag(torch.ones(dim))
    batch_diag_mask = diag_mask.unsqueeze(0).expand(batch_size, dim, dim)
    batch_diag_values = diag_values.unsqueeze(1).expand(batch_size, dim, dim)
    return batch_diag_values * batch_diag_mask

def batch_square_vector(vector, M):
    """
    Compute x^T M x
    """
    vector = vector.unsqueeze(2)
    return torch.bmm(torch.bmm(vector.transpose(2, 1), M), vector).squeeze(2)

def fanin_init(tensor):
    size = tensor.size()
    if len(size) == 2:
        fan_in = size[0]
    elif len(size) > 2:
        fan_in = np.prod(size[1:])
    else:
        raise Exception("Shape must be have dimension at least 2.")
    bound = 1. / np.sqrt(fan_in)
    return tensor.data.uniform_(-bound, bound)

def fanin_init_weights_like(tensor):
    size = tensor.size()
    if len(size) == 2:
        fan_in = size[0]
    elif len(size) > 2:
        fan_in = np.prod(size[1:])
    else:
        raise Exception("Shape must be have dimension at least 2.")
    bound = 1. / np.sqrt(fan_in)
    new_tensor = FloatTensor(tensor.size())
    new_tensor.uniform_(-bound, bound)
    return new_tensor

def almost_identity_weights_like(tensor):
    """
    Set W = I + lambda * Gaussian no
    :param tensor:
    :return:
    """
    shape = tensor.size()
    init_value = np.eye(*shape)
    init_value += 0.01 * np.random.rand(*shape)
    return FloatTensor(init_value)

def clip1(x):
    return torch.clamp(x, -1, 1)

def compute_conv_output_size(h_in, w_in, kernel_size, stride, padding=0):
    h_out = (h_in + 2 * padding - (kernel_size-1) - 1)/stride + 1
    w_out = (w_in + 2 * padding - (kernel_size-1) - 1)/stride + 1
    return int(np.floor(h_out)), int(np.floor(w_out))

def compute_deconv_output_size(h_in, w_in, kernel_size, stride, padding=0):
    h_out = (h_in -1)*stride - 2*padding + kernel_size
    w_out = (w_in -1)*stride - 2*padding + kernel_size
    return int(np.floor(h_out)), int(np.floor(w_out))

def compute_conv_layer_sizes(h_in, w_in, kernel_sizes, strides, paddings=None):
    if paddings==None:
        for kernel, stride in zip(kernel_sizes, strides):
            h_in, w_in = compute_conv_output_size(h_in, w_in, kernel, stride)
            print('Output Size:', (h_in, w_in))
    else:
        for kernel, stride, padding in zip(kernel_sizes, strides, paddings):
            h_in, w_in = compute_conv_output_size(h_in, w_in, kernel, stride, padding=padding)
            print('Output Size:', (h_in, w_in))

def compute_deconv_layer_sizes(h_in, w_in, kernel_sizes, strides, paddings=None):
    if paddings==None:
        for kernel, stride in zip(kernel_sizes, strides):
            h_in, w_in = compute_deconv_output_size(h_in, w_in, kernel, stride)
            print('Output Size:', (h_in, w_in))
    else:
        for kernel, stride, padding in zip(kernel_sizes, strides, paddings):
            h_in, w_in = compute_deconv_output_size(h_in, w_in, kernel, stride, padding=padding)
            print('Output Size:', (h_in, w_in))

"""
GPU wrappers
"""

_use_gpu = False
device = None

def set_gpu_mode(mode, gpu_id=0):
    global _use_gpu
    global device
    global _gpu_id
    _gpu_id = gpu_id
    _use_gpu = mode
    device = torch.device("cuda:"+str(gpu_id) if _use_gpu else "cpu")

def gpu_enabled():
    return _use_gpu

def set_device(gpu_id):
    torch.cuda.set_device(gpu_id)

# noinspection PyPep8Naming
def FloatTensor(*args, torch_device=None, **kwargs):
    if torch_device is None:
        torch_device = device
    return torch.FloatTensor(*args, **kwargs, device=torch_device)

def from_numpy(*args, **kwargs):
    return torch.from_numpy(*args, **kwargs).float().to(device)

def from_numpy_or_pytorch(arr):
    if isinstance(arr, torch.Tensor):
        return arr
    else:
        return torch.from_numpy(arr).float().to(device)

def image_from_numpy(np_arr, *args, **kwargs):
    dtype = np_arr.dtype
    pt_arr = from_numpy(np.ascontiguousarray(np_arr), *args, **kwargs)
    if dtype == np.uint8: return pt_arr / 255.
    else: return pt_arr
    
def get_numpy(inst):
    if isinstance(inst, (np.ndarray, int, float)):
        return inst
    elif isinstance(inst, dict):
        return {k: get_numpy(v) for k, v in inst.items()}
    elif isinstance(inst, torch.Tensor):
        return inst.to('cpu').detach().numpy()
    else:
        raise ValueError(f"Cannot convert data of type: {type(inst)} to numpy")

# "get" is a weird name for a converter. "to" is better.
to_numpy = get_numpy

def zeros(*sizes, torch_device=None, **kwargs):
    if torch_device is None:
        torch_device = device
    return torch.zeros(*sizes, **kwargs, device=torch_device)

def ones(*sizes, torch_device=None, **kwargs):
    if torch_device is None:
        torch_device = device
    return torch.ones(*sizes, **kwargs, device=torch_device)

def ones_like(*args, torch_device=None, **kwargs):
    if torch_device is None:
        torch_device = device
    return torch.ones_like(*args, **kwargs, device=torch_device)

def randn(*args, torch_device=None, **kwargs):
    if torch_device is None:
        torch_device = device
    return torch.randn(*args, **kwargs, device=torch_device)

def zeros_like(*args, torch_device=None, **kwargs):
    if torch_device is None:
        torch_device = device
    return torch.zeros_like(*args, **kwargs, device=torch_device)

def tensor(*args, torch_device=None, **kwargs):
    if torch_device is None:
        torch_device = device
    return torch.tensor(*args, **kwargs, device=torch_device)

def choose_best_gpu(use=True):
    avail = torch.cuda.is_available() and use
    if not avail:
        set_gpu_mode(False)
        return
    
    # # Amount free on each gpu.
    # try:
    #     free_space_per_gpu = gpu_util.NVLog().get_free_space_per_gpu()
    #     if len(os.environ.get('CUDA_VISIBLE_DEVICES', '')) > 0:
    #         print(f"free_space_per_gpu {free_space_per_gpu}")
    #         print(f"CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']}")
    #         subset_available = list(sorted([int(_) for _ in os.environ['CUDA_VISIBLE_DEVICES'].split(',')]))
    #         n_missing = len(free_space_per_gpu) - len(subset_available)
    #         print(f"gpus available={subset_available}")
    #         free_space_per_gpu = [_ for idx, _ in enumerate(free_space_per_gpu) if idx in subset_available]
    #         print(f"free_space_per_gpu={free_space_per_gpu}")
    #     else:
    #         n_missing = 0
    #         subset_available = list(range(len(free_space_per_gpu)))
    #     most_free_idx = np.argmax(free_space_per_gpu)
    #     most_free_gpu_idx = subset_available[most_free_idx]
    # except:
    #     print("Getting gpu info failed. Using gpu 0")
    #     most_free_gpu_idx = 0
    #     most_free_idx = 0

    most_free_idx = gpu_util.get_freest_gpu_id()
    print(f"Using gpu={most_free_idx}")
    log.info(f"Using gpu={most_free_idx}")
    set_gpu_mode(avail, most_free_idx)
    
def iter_graph(root, callback):
    queue = [root]
    seen = set()
    while queue:
        fn = queue.pop()
        if fn in seen:
            continue
        seen.add(fn)
        for next_fn, _ in fn.next_functions:
            if next_fn is not None:
                queue.append(next_fn)
        callback(fn)

def register_hooks(var):
    fn_dict = {}
    def hook_cb(fn):
        def register_grad(grad_input, grad_output):
            fn_dict[fn] = grad_input
        fn.register_hook(register_grad)
    iter_graph(var.grad_fn, hook_cb)

    def is_bad_grad(grad_output):
        if grad_output is None:
            pdb.set_trace()
        grad_output = grad_output.data
        return grad_output.ne(grad_output).any() or grad_output.gt(1e6).any()

    def make_dot():
        node_attr = dict(style='filled',
                         shape='box',
                         align='left',
                         fontsize='12',
                         ranksep='0.1',
                         height='0.2')
        dot = Digraph(node_attr=node_attr, graph_attr=dict(size="12,12"))

        def size_to_str(size):
            return '('+(', ').join(map(str, size))+')'

        def build_graph(fn):
            if hasattr(fn, 'variable'):  # if GradAccumulator
                u = fn.variable
                node_name = 'Variable\n ' + size_to_str(u.size())
                dot.node(str(id(u)), node_name, fillcolor='lightblue')
            else:
                assert fn in fn_dict, fn
                fillcolor = 'white'
                if any(is_bad_grad(gi) for gi in fn_dict[fn]):
                    fillcolor = 'red'
                dot.node(str(id(fn)), str(type(fn).__name__), fillcolor=fillcolor)
            for next_fn, _ in fn.next_functions:
                if next_fn is not None:
                    next_id = id(getattr(next_fn, 'variable', next_fn))
                    dot.edge(str(next_id), str(id(fn)))
        iter_graph(var.grad_fn, build_graph)

        return dot

    return make_dot
    

import abc

import numpy as np
import torch
from torch import nn as nn

class PyTorchModule(nn.Module, metaclass=abc.ABCMeta):
    """
    Keeping wrapper around to be a bit more future-proof.
    """
    pass


def eval_np(module, *args, **kwargs):
    """
    Eval this module with a numpy interface

    Same as a call to __call__ except all Variable input/outputs are
    replaced with numpy equivalents.

    Assumes the output is either a single object or a tuple of objects.
    """
    torch_args = tuple(torch_ify(x) for x in args)
    torch_kwargs = {k: torch_ify(v) for k, v in kwargs.items()}
    outputs = module(*torch_args, **torch_kwargs)
    if isinstance(outputs, tuple):
        return tuple(np_ify(x) for x in outputs)
    else:
        return np_ify(outputs)


def torch_ify(np_array_or_other):
    if isinstance(np_array_or_other, np.ndarray):
        return from_numpy(np_array_or_other)
    else:
        return np_array_or_other


def np_ify(tensor_or_other):
    if isinstance(tensor_or_other, torch.autograd.Variable):
        return get_numpy(tensor_or_other)
    else:
        return tensor_or_other


def _elem_or_tuple_to_variable(elem_or_tuple):
    if isinstance(elem_or_tuple, tuple):
        return tuple(
            _elem_or_tuple_to_variable(e) for e in elem_or_tuple
        )
    return from_numpy(elem_or_tuple).float()


def _filter_batch(np_batch):
    for k, v in np_batch.items():
        if hasattr(v, 'dtype') and v.dtype == np.bool:
            yield k, v.astype(int)
        else:
            yield k, v


def np_to_pytorch_batch(np_batch):
    return {
        k: _elem_or_tuple_to_variable(x)
        for k, x in _filter_batch(np_batch)
        if x is not None and x.dtype != np.dtype('O')  # ignore object (e.g. dictionaries)
    }
