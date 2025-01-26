import torch
import torch.nn as nn
import re
import copy


class LayerNorm2d(nn.LayerNorm):
    rmsnorm = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x if LayerNorm2d.rmsnorm else x - torch.mean(x, dim=1, keepdim=True)
        out = out / torch.sqrt(torch.square(out).mean(dim=1, keepdim=True) + self.eps)
        if self.elementwise_affine:
            out = out * self.weight.view(1, -1, 1, 1) + self.bias.view(1, -1, 1, 1)
        return out

    def extra_repr(self) -> str:
        return f"{self.normalized_shape}, eps={self.eps}, elementwise_affine={self.elementwise_affine}, rmsnorm={self.rmsnorm}"


REGISTERED_NORMALIZATION_DICT: dict[str, tuple[type, dict[str, any]]] = {
    "bn2d": (
        nn.BatchNorm2d,
        {"num_features": None, "eps": 1e-5, "momentum": 0.1, "affine": True},
    ),
    "syncbn": (
        nn.SyncBatchNorm,
        {"num_features": None, "eps": 1e-5, "momentum": 0.1, "affine": True},
    ),
    "ln": (
        nn.LayerNorm,
        {"normalized_shape": None, "eps": 1e-5, "elementwise_affine": True},
    ),
    "ln2d": (
        LayerNorm2d,
        {"normalized_shape": None, "eps": 1e-5, "elementwise_affine": True},
    ),
}

REGISTERED_ACT_DICT: dict[str, tuple[type, dict[str, any]]] = {
    "relu": (nn.ReLU, {"inplace": True}),
    "relu6": (nn.ReLU6, {"inplace": True}),
    "hswish": (nn.Hardswish, {"inplace": True}),
    "hsigmoid": (nn.Hardsigmoid, {"inplace": True}),
    "swish": (nn.SiLU, {"inplace": True}),
    "silu": (nn.SiLU, {"inplace": True}),
    "tanh": (nn.Tanh, {}),
    "sigmoid": (nn.Sigmoid, {}),
    "gelu": (nn.GELU, {"approximate": "tanh"}),
    "mish": (nn.Mish, {"inplace": True}),
    "identity": (nn.Identity, {}),
}


class IdentityMap(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, *args, **kwargs):
        return x

    @property
    def config(self):
        return {"mm_projector_type": "identity"}


class SimpleResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.pre_norm = nn.LayerNorm(channels)

        self.proj = nn.Sequential(
            nn.Linear(channels, channels), nn.GELU(), nn.Linear(channels, channels)
        )

    def forward(self, x):
        x = self.pre_norm(x)
        return x + self.proj(x)


def build_act(name: str or None, **kwargs) -> nn.Module or None:
    if name in REGISTERED_ACT_DICT:
        act_cls, default_args = copy.deepcopy(REGISTERED_ACT_DICT[name])
        for key in default_args:
            if key in kwargs:
                default_args[key] = kwargs[key]
        return act_cls(**default_args)
    elif name is None or name.lower() == "none":
        return None
    else:
        raise ValueError(f"do not support: {name}")


def build_norm(
    name="bn2d", num_features=None, affine=True, **kwargs
) -> nn.Module or None:
    if name in ["ln", "ln2d"]:
        kwargs["normalized_shape"] = num_features
        kwargs["elementwise_affine"] = affine
    else:
        kwargs["num_features"] = num_features
        kwargs["affine"] = affine
    if name in REGISTERED_NORMALIZATION_DICT:
        norm_cls, default_args = copy.deepcopy(REGISTERED_NORMALIZATION_DICT[name])
        for key in default_args:
            if key in kwargs:
                default_args[key] = kwargs[key]
        return norm_cls(**default_args)
    elif name is None or name.lower() == "none":
        return None
    else:
        raise ValueError("do not support: %s" % name)


def get_same_padding(kernel_size: int or tuple[int, ...]) -> int or tuple[int, ...]:
    if isinstance(kernel_size, tuple):
        return tuple([get_same_padding(ks) for ks in kernel_size])
    else:
        assert kernel_size % 2 > 0, f"kernel size {kernel_size} should be odd number"
        return kernel_size // 2


def val2list(x: list or tuple or any, repeat_time=1) -> list:  # type: ignore
    """Repeat `val` for `repeat_time` times and return the list or val if list/tuple."""
    if isinstance(x, (list, tuple)):
        return list(x)
    return [x for _ in range(repeat_time)]


def val2tuple(x: list or tuple or any, min_len: int = 1, idx_repeat: int = -1) -> tuple:  # type: ignore
    """Return tuple with min_len by repeating element at idx_repeat."""
    # convert to list first
    x = val2list(x)

    # repeat elements if necessary
    if len(x) > 0:
        x[idx_repeat:idx_repeat] = [x[idx_repeat] for _ in range(min_len - len(x))]

    return tuple(x)


class ConvLayer(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        kernel_size=3,
        stride=1,
        dilation=1,
        groups=1,
        padding: int or None = None,
        use_bias=False,
        dropout=0.0,
        norm="bn2d",
        act="relu",
    ):
        super().__init__()
        if padding is None:
            padding = get_same_padding(kernel_size)
            padding *= dilation

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.groups = groups
        self.padding = padding
        self.use_bias = use_bias

        self.dropout = nn.Dropout2d(dropout, inplace=False) if dropout > 0 else None
        self.conv = nn.Conv2d(
            in_dim,
            out_dim,
            kernel_size=(kernel_size, kernel_size),
            stride=(stride, stride),
            padding=padding,
            dilation=(dilation, dilation),
            groups=groups,
            bias=use_bias,
        )
        self.norm = build_norm(norm, num_features=out_dim)
        self.act = build_act(act)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.dropout is not None:
            x = self.dropout(x)
        x = self.conv(x)
        if self.norm:
            x = self.norm(x)
        if self.act:
            x = self.act(x)
        return x


class GLUMBConv(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        out_feature=None,
        kernel_size=3,
        stride=1,
        padding: int or None = None,
        use_bias=False,
        norm=(None, None, None),
        act=("silu", "silu", None),
        dilation=1,
    ):
        self.out_feature = out_feature or in_features
        super().__init__()
        use_bias = val2tuple(use_bias, 3)
        norm = val2tuple(norm, 3)
        act = val2tuple(act, 3)

        self.glu_act = build_act(act[1], inplace=False)
        self.inverted_conv = ConvLayer(
            in_features,
            hidden_features * 2,
            1,
            use_bias=use_bias[0],
            norm=norm[0],
            act=act[0],
        )
        self.depth_conv = ConvLayer(
            hidden_features * 2,
            hidden_features * 2,
            kernel_size,
            stride=stride,
            groups=hidden_features * 2,
            padding=padding,
            use_bias=use_bias[1],
            norm=norm[1],
            act=None,
            dilation=dilation,
        )
        self.point_conv = ConvLayer(
            hidden_features,
            out_feature,
            1,
            use_bias=use_bias[2],
            norm=norm[2],
            act=act[2],
        )
        # from IPython import embed; embed(header='debug dilate conv')

    def forward(self, x: torch.Tensor, HW=None) -> torch.Tensor:
        B, N, C = x.shape
        if HW is None:
            H = W = int(N**0.5)
        else:
            H, W = HW

        x = x.reshape(B, H, W, C).permute(0, 3, 1, 2)
        x = self.inverted_conv(x)
        x = self.depth_conv(x)

        x, gate = torch.chunk(x, 2, dim=1)
        gate = self.glu_act(gate)
        x = x * gate

        x = self.point_conv(x)
        x = x.reshape(B, self.out_feature, N).permute(0, 2, 1)

        return x


class DWMlp(nn.Module):
    """MLP as used in Vision Transformer, MLP-Mixer and related networks"""

    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        bias=True,
        drop=0.0,
        kernel_size=3,
        stride=1,
        dilation=1,
        padding=None,
    ):
        super().__init__()
        hidden_features = hidden_features or in_features
        self.hidden_features = hidden_features
        if padding is None:
            padding = get_same_padding(kernel_size)
            padding *= dilation

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.conv = nn.Conv2d(
            hidden_features,
            hidden_features,
            kernel_size=(kernel_size, kernel_size),
            stride=(stride, stride),
            padding=padding,
            dilation=(dilation, dilation),
            groups=hidden_features,
            bias=bias,
        )
        self.fc2 = nn.Linear(hidden_features, hidden_features)

    def forward(self, x, HW=None):
        B, N, C = x.shape
        if HW is None:
            H = W = int(N**0.5)
        else:
            H, W = HW
        x = self.fc1(x)
        x = self.act(x)
        x = x.reshape(B, H, W, self.hidden_features).permute(0, 3, 1, 2)
        x = self.conv(x)
        x = x.reshape(B, self.hidden_features, N).permute(0, 2, 1)
        x = self.fc2(x)
        return x


def build_vision_projector(config, delay_load=False, **kwargs):
    projector_type = getattr(config, "mm_projector_type", "linear")

    if projector_type == "linear":
        return nn.Linear(config.mm_hidden_size, config.hidden_size)

    mlp_gelu_match = re.match(r"^mlp(\d+)x_gelu$", projector_type)

    # mm_hidden_size: 1024, hidden_size: 4096
    if mlp_gelu_match:
        mlp_depth = int(mlp_gelu_match.group(1))
        modules = [nn.Linear(config.mm_hidden_size, config.hidden_size)]
        for _ in range(1, mlp_depth):
            modules.append(nn.GELU())
            modules.append(nn.Linear(config.hidden_size, config.hidden_size))
        # print(
        #     "################### USING mlp2x_gelu!!!!!!!!!!!!!!!!!!!!!!!! ###################"
        # )
        return nn.Sequential(*modules)
    elif projector_type == "glumbconv1x_dilate2x":
        modules = [
            GLUMBConv(
                in_features=config.mm_hidden_size,
                hidden_features=config.hidden_size,
                out_feature=config.hidden_size,
                use_bias=(True, True, False),
                norm=(None, None, None),
                act=("silu", "silu", None),
                dilation=2,
            )
        ]
        # print(
        #     "################### USING GLUMBCONV1X!!!!!!!!!!!!!!!!!!!!!!!! ###################"
        # )
        return nn.Sequential(*modules)
    elif projector_type == "glumbconv1x":
        modules = [
            GLUMBConv(
                in_features=config.mm_hidden_size,
                hidden_features=config.hidden_size,
                out_feature=config.hidden_size,
                use_bias=(True, True, False),
                norm=(None, None, None),
                act=("silu", "silu", None),
            )
        ]
        return nn.Sequential(*modules)
    elif projector_type == "dwconv1x":
        modules = [
            DWMlp(
                in_features=config.mm_hidden_size,
                hidden_features=config.hidden_size,
                drop=0,
            )
        ]
        return nn.Sequential(*modules)

    if projector_type == "identity":
        return IdentityMap()

    raise ValueError(f"Unknown projector type: {projector_type}")
