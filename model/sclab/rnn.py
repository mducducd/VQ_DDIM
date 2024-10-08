
import torch
import torch.nn as nn
from model.belfusion.torch import *
from einops import rearrange
import math
"""
Finite Scalar Quantization: VQ-VAE Made Simple - https://arxiv.org/abs/2309.15505
Code adapted from Jax version in Appendix A.1
"""

from typing import List, Optional

import torch
import torch.nn as nn
from einops import pack, rearrange, unpack
from torch import Tensor, int32
from torch.cuda.amp import autocast
from torch.nn import Module

# helper functions


def exists(v):
    return v is not None


def default(*args):
    for arg in args:
        if exists(arg):
            return arg
    return None


def pack_one(t, pattern):
    return pack([t], pattern)


def unpack_one(t, ps, pattern):
    return unpack(t, ps, pattern)[0]


# tensor helpers


def round_ste(z: Tensor) -> Tensor:
    """Round with straight through gradients."""
    zhat = z.round()
    return z + (zhat - z).detach()


# main class


class FSQ(Module):
    def __init__(
        self,
        levels: List[int],
        dim: Optional[int] = None,
        num_codebooks=1,
        keep_num_codebooks_dim: Optional[bool] = None,
        scale: Optional[float] = None,
    ):
        super().__init__()
        _levels = torch.tensor(levels, dtype=int32)
        self.register_buffer("_levels", _levels, persistent=False)

        _basis = torch.cumprod(torch.tensor([1] + levels[:-1]), dim=0, dtype=int32)
        self.register_buffer("_basis", _basis, persistent=False)

        self.scale = scale

        codebook_dim = len(levels)
        self.codebook_dim = codebook_dim

        effective_codebook_dim = codebook_dim * num_codebooks
        self.num_codebooks = num_codebooks
        self.effective_codebook_dim = effective_codebook_dim

        keep_num_codebooks_dim = default(keep_num_codebooks_dim, num_codebooks > 1)
        assert not (num_codebooks > 1 and not keep_num_codebooks_dim)
        self.keep_num_codebooks_dim = keep_num_codebooks_dim

        self.dim = default(dim, len(_levels) * num_codebooks)

        has_projections = self.dim != effective_codebook_dim
        self.project_in = (
            nn.Linear(self.dim, effective_codebook_dim)
            if has_projections
            else nn.Identity()
        )
        self.project_out = (
            nn.Linear(effective_codebook_dim, self.dim)
            if has_projections
            else nn.Identity()
        )
        self.has_projections = has_projections

        self.codebook_size = self._levels.prod().item()

        implicit_codebook = self.indices_to_codes(
            torch.arange(self.codebook_size), project_out=False
        )
        self.register_buffer("implicit_codebook", implicit_codebook, persistent=False)

    def bound(self, z: Tensor, eps: float = 1e-3) -> Tensor:
        """Bound `z`, an array of shape (..., d)."""
        half_l = (self._levels - 1) * (1 + eps) / 2
        offset = torch.where(self._levels % 2 == 0, 0.5, 0.0)
        shift = (offset / half_l).atanh()
        return (z + shift).tanh() * half_l - offset

    def quantize(self, z: Tensor) -> Tensor:
        """Quantizes z, returns quantized zhat, same shape as z."""
        quantized = round_ste(self.bound(z))
        half_width = self._levels // 2  # Renormalize to [-1, 1].
        return quantized / half_width

    def _scale_and_shift(self, zhat_normalized: Tensor) -> Tensor:
        half_width = self._levels // 2
        return (zhat_normalized * half_width) + half_width

    def _scale_and_shift_inverse(self, zhat: Tensor) -> Tensor:
        half_width = self._levels // 2
        return (zhat - half_width) / half_width

    def codes_to_indices(self, zhat: Tensor) -> Tensor:
        """Converts a `code` to an index in the codebook."""
        assert zhat.shape[-1] == self.codebook_dim
        zhat = self._scale_and_shift(zhat)
        return (zhat * self._basis).sum(dim=-1).to(int32)

    def indices_to_codes(self, indices: Tensor, project_out=True) -> Tensor:
        """Inverse of `codes_to_indices`."""

        is_img_or_video = indices.ndim >= (3 + int(self.keep_num_codebooks_dim))

        indices = rearrange(indices, "... -> ... 1")
        codes_non_centered = (indices // self._basis) % self._levels
        codes = self._scale_and_shift_inverse(codes_non_centered)

        if self.keep_num_codebooks_dim:
            codes = rearrange(codes, "... c d -> ... (c d)")

        if project_out:
            codes = self.project_out(codes)

        if is_img_or_video:
            codes = rearrange(codes, "b ... d -> b d ...")

        return codes

    @autocast(enabled=False)
    def forward(self, z: Tensor) -> Tensor:
        """
        einstein notation
        b - batch
        n - sequence (or flattened spatial dimensions)
        d - feature dimension, which is also log2(codebook size)
        c - number of codebook dim
        """

        is_img_or_video = z.ndim >= 4

        # standardize image or video into (batch, seq, dimension)

        if is_img_or_video:
            z = rearrange(z, "b d ... -> b ... d")
            z, ps = pack_one(z, "b * d")

        assert (
            z.shape[-1] == self.dim
        ), f"expected dimension of {self.dim} but found dimension of {z.shape[-1]}"

        z = self.project_in(z)

        z = rearrange(z, "b n (c d) -> b n c d", c=self.num_codebooks)

        codes = self.quantize(z)
        indices = self.codes_to_indices(codes)

        codes = rearrange(codes, "b n c d -> b n (c d)")

        out = self.project_out(codes)

        # reconstitute image or video dimensions

        if is_img_or_video:
            out = unpack_one(out, ps, "b * d")
            out = rearrange(out, "b ... d -> b d ...")

            indices = unpack_one(indices, ps, "b * c")

        if not self.keep_num_codebooks_dim:
            indices = rearrange(indices, "... 1 -> ...")

        return out, indices
nl = {
    "tanh": nn.Tanh,
    "relu": nn.ReLU,
    "sigmoid": nn.Sigmoid,
    "elu": nn.ELU,
    "selu": nn.SELU,
    "softplus": nn.Softplus,
    "softsign": nn.Softsign,
    "leaky_relu": nn.LeakyReLU,
    "none": lambda x: x,
}


def sample(mu):
    return torch.randn_like(mu)

def rc(x_start, pred, batch_first=True):
    # x_start -> [batch_size, ...]
    # pred -> [seq_length, batch_size, ...] | [batch_size, seq_length, ...]
    if batch_first:
        x_start = x_start.unsqueeze(1)
        shapes = [1 for s in x_start.shape]
        shapes[1] = pred.shape[1]
        x_start = x_start.repeat(shapes)
    else:
        x_start = x_start.unsqueeze(0)
        shapes = [1 for s in x_start.shape]
        shapes[0] = pred.shape[0]
        x_start = x_start.repeat(shapes)
    return x_start + pred

def rc_recurrent(x_start, pred, batch_first=True): # residual connection => offsets modeling
    # x_start -> [batch_size, ...]
    # pred -> [seq_length, batch_size, ...] | [batch_size, seq_length, ...]
    if batch_first:
        pred[:, 0] = x_start + pred[:, 0]
        for i in range(1, pred.shape[1]):
            pred[:, i] = pred[:, i-1] + pred[:, i]
    else: # seq length first
        pred[0] = x_start + pred[0]
        for i in range(1, pred.shape[0]):
            pred[i] = pred[i-1] + pred[i]
    return pred

class BasicMLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dims=[], dropout=0.5, non_linearities='relu'):
        super(BasicMLP, self).__init__()
        self.non_linearities = non_linearities

        self.dropout = nn.Dropout(dropout)
        self.nl = nl[non_linearities]()

        self.denses = None
        
        # hidden dims
        hidden_dims = hidden_dims + [output_dim, ] # output dim is treated as the last hidden dim

        seqs = []
        for i in range(len(hidden_dims)):
            linear = nn.Linear(input_dim if i==0 else hidden_dims[i-1], hidden_dims[i])
            init_weights(linear)
            seqs.append(nn.Sequential(self.dropout, linear, self.nl))

        self.denses = nn.Sequential(*seqs)

    def forward(self, x):
        return self.denses(x) if self.denses is not None else x


class MLP(nn.Module):
    # https://github.com/Khrylx/DLow
    def __init__(self, input_dim, hidden_dims=(128, 128), activation='tanh'):
        super(MLP, self).__init__()
        if activation == 'tanh':
            self.activation = torch.tanh
        elif activation == 'relu':
            self.activation = torch.relu
        elif activation == 'sigmoid':
            self.activation = torch.sigmoid

        self.out_dim = hidden_dims[-1]
        self.affine_layers = nn.ModuleList()
        last_dim = input_dim
        for nh in hidden_dims:
            self.affine_layers.append(nn.Linear(last_dim, nh))
            last_dim = nh

    def forward(self, x):
        for affine in self.affine_layers:
            x = self.activation(affine(x))
        return x

class RNN(nn.Module):
    # https://github.com/Khrylx/DLow
    def __init__(self, input_dim, out_dim, cell_type='lstm', bi_dir=False):
        super(RNN, self).__init__()
        self.input_dim = input_dim
        self.out_dim = out_dim
        self.cell_type = cell_type
        self.bi_dir = bi_dir
        self.mode = 'batch'
        rnn_cls = nn.LSTMCell if cell_type == 'lstm' else nn.GRUCell
        hidden_dim = out_dim // 2 if bi_dir else out_dim
        self.rnn_f = rnn_cls(self.input_dim, hidden_dim)
        if bi_dir:
            self.rnn_b = rnn_cls(self.input_dim, hidden_dim)
        self.hx, self.cx = None, None

    def set_mode(self, mode):
        self.mode = mode

    def initialize(self, batch_size=1, hx=None, cx=None):
        if self.mode == 'step':
            self.hx = zeros((batch_size, self.rnn_f.hidden_size)) if hx is None else hx
            if self.cell_type == 'lstm':
                self.cx = zeros((batch_size, self.rnn_f.hidden_size)) if cx is None else cx

    def forward(self, x):
        if self.mode == 'step':
            self.hx, self.cx = batch_to(x.device, self.hx, self.cx)
            if self.cell_type == 'lstm':
                self.hx, self.cx = self.rnn_f(x, (self.hx, self.cx))
            else:
                self.hx = self.rnn_f(x, self.hx)
            rnn_out = self.hx
        else:
            rnn_out_f = self.batch_forward(x)
            if not self.bi_dir:
                return rnn_out_f
            rnn_out_b = self.batch_forward(x, reverse=True)
            rnn_out = torch.cat((rnn_out_f, rnn_out_b), 2)
        return rnn_out

    def batch_forward(self, x, reverse=False):
        rnn = self.rnn_b if reverse else self.rnn_f
        rnn_out = []
        hx = zeros((x.size(1), rnn.hidden_size), device=x.device)
        if self.cell_type == 'lstm':
            cx = zeros((x.size(1), rnn.hidden_size), device=x.device)
        ind = reversed(range(x.size(0))) if reverse else range(x.size(0))
        for t in ind:
            if self.cell_type == 'lstm':
                hx, cx = rnn(x[t, ...], (hx, cx))
            else:
                hx = rnn(x[t, ...], hx)
            rnn_out.append(hx.unsqueeze(0))
        if reverse:
            rnn_out.reverse()
        rnn_out = torch.cat(rnn_out, 0)
        return rnn_out



class AutoencoderRNN_VAE_v2(nn.Module):
    def __init__(self, cfg):
        super(AutoencoderRNN_VAE_v2, self).__init__()
        self.seq_len = cfg.seq_len
        self.window_size = cfg.window_size
        assert self.seq_len % self.window_size == 0, "seq_len must be divisible by window_size"
        self.hidden_dim = cfg.hidden_dim
        self.z_dim = cfg.z_dim
        self.emb_dims = cfg.emb_dims
        self.num_layers = cfg.num_layers
        self.rnn_type = cfg.rnn_type
        self.dropout = cfg.dropout
        self.emotion_dim = cfg.emotion_dim
        self.coeff_3dmm_dim = cfg.coeff_3dmm_dim

        # encode
        self.x_rnn = RNN(self.emotion_dim, self.hidden_dim, cell_type=self.rnn_type)
        # z
        self.fc_mu_enc = nn.Linear(self.hidden_dim, self.z_dim)
        self.fc_logvar_enc = nn.Linear(self.hidden_dim, self.z_dim)
        
        # decode
        self.fc_z_dec = nn.Linear(self.z_dim, self.hidden_dim)
        self.d_rnn = RNN(self.emotion_dim + self.hidden_dim, self.hidden_dim, cell_type=self.rnn_type)
        self.d_mlp = MLP(self.hidden_dim, self.emb_dims)
        self.d_out = nn.Linear(self.d_mlp.out_dim, self.emotion_dim)
        self.d_rnn.set_mode('step')
        # decode_3dmm
        # linear layers from emotion_dim to coeff_3dmm_dim
        self.coeff_reg = nn.Sequential(
            nn.Linear(self.emotion_dim, self.hidden_dim),
            nn.Tanh(),
            nn.Linear(self.hidden_dim, self.coeff_3dmm_dim),
        )

        self.dropout = nn.Dropout(self.dropout)

    def _encode(self, x):
        assert x.shape[0] == self.window_size # otherwise it does not make sense
        return self.x_rnn(x)[-1]
    
    def _decode(self, h_y):
        h_y = self.fc_z_dec(h_y)
        h_y = self.dropout(h_y)
        self.d_rnn.initialize(batch_size=h_y.shape[0])
        y = []
        for i in range(self.window_size):
            y_p = torch.zeros((h_y.shape[0], self.emotion_dim), device=h_y.device) if i == 0 else y_i
            rnn_in = torch.cat([h_y, y_p], dim=1)
            h = self.d_rnn(rnn_in)
            h = self.d_mlp(h)
            y_i = self.d_out(h)
            y.append(y_i)

        return torch.stack(y)
    
    def encode_all(self, emotion_seq):
        # emotion_seq: (batch_size, seq_len, emotion_dim)
        # we encode the emotion_seq in windows of size window_size
        batch_size, seq_len = emotion_seq.shape[:2]
        assert seq_len % self.window_size == 0, "seq_len must be divisible by window_size"
        emotion_seq = emotion_seq.reshape(batch_size * (seq_len // self.window_size), self.window_size, -1)
        # as a result, we have the (seq_len // window_length) windows for each batch element, sequentially

        emotion_seq = rearrange(emotion_seq, 'b s f -> s b f' )

        # original code from DLow repository
        h_x = self._encode(emotion_seq) # [batch_size * num_windows, features_size]
        return h_x
    
    def encode(self, emotion_seq):
        # if seq_len > self.window_size --> sample a random window from emotion_seq
        # if seq_len == self.window_size --> encode the whole emotion_seq
        batch_size, seq_len = emotion_seq.shape[:2]
        
        if seq_len > self.window_size:
            selected_window = np.random.randint(0, seq_len // self.window_size)
            selected_emotion_seq = emotion_seq[:, selected_window * self.window_size : (selected_window + 1) * self.window_size, :]
        elif seq_len == self.window_size:
            selected_emotion_seq = emotion_seq
        else:
            raise ValueError("seq_len must be at least window_size")
        
        return self._encode(rearrange(selected_emotion_seq, 'b s f -> s b f' ))

    def decode(self, emb):
        Y_r = self._decode(emb)

        # BACK TO ORIGINAL SHAPE
        Y_r = rearrange(Y_r, "s b f -> b s f")
        return Y_r
    
    def decode_coeff(self, emotion):
        return self.coeff_reg(emotion)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, listener_emotion=None, listener_3dmm=None, **kwargs):
        """
        video: ...
        audio: ...
        speaker_emotion: (batch_size, seq_len (750), emotion_dim (25))
        """
        batch_size, frame_num = listener_emotion.shape[:2]

        # stack window_size frames together
        # from [batch_size, seq_len, emotion_dim] to [batch_size * (seq_len // window_length), window_length, emotion_dim]
        emotion = listener_emotion.reshape(batch_size * (frame_num // self.window_size), self.window_size, -1)
        # as a result, we have the (seq_len // window_length) windows for each batch element, sequentially

        emotion = rearrange(emotion, 'b s f -> s b f' )

        # original code from DLow repository
        h_x = self._encode(emotion)
        mu = self.fc_mu_enc(h_x)
        logvar = self.fc_logvar_enc(h_x)
        if self.training:
            z = self.reparameterize(mu, logvar)
        else:
            z = mu

        Y_r = self._decode(z)
        # BACK TO ORIGINAL SHAPE
        Y_r = rearrange(Y_r, "s b f -> b s f")
        Y_r = Y_r.reshape(batch_size, frame_num, -1) # undo the window-size stacking

        return {
            "prediction": Y_r,
            "target": listener_emotion, # we are autoencoding
            "coefficients_3dmm": self.decode_coeff(Y_r), 
            "target_coefficients": listener_3dmm, 
            "mu": mu, 
            "logvar": logvar, 
        }

class AutoencoderRNN_VQVAE_v2(nn.Module):
    def __init__(self, cfg):
        super(AutoencoderRNN_VQVAE_v2, self).__init__()
        self.seq_len = cfg.seq_len
        self.window_size = cfg.window_size
        assert self.seq_len % self.window_size == 0, "seq_len must be divisible by window_size"
        self.hidden_dim = cfg.hidden_dim
        self.z_dim = cfg.z_dim
        self.emb_dims = cfg.emb_dims
        self.num_layers = cfg.num_layers
        self.rnn_type =  cfg.rnn_type
        self.dropout = cfg.dropout
        self.emotion_dim = cfg.emotion_dim
        self.coeff_3dmm_dim = cfg.coeff_3dmm_dim

        # encode
        self.x_rnn = RNN(self.emotion_dim, self.hidden_dim, cell_type=self.rnn_type)
        # z
        # self.fc_mu_enc = nn.Linear(self.hidden_dim, self.z_dim)
        # self.fc_logvar_enc = nn.Linear(self.hidden_dim, self.z_dim)
        
        # decode
        self.fc_z_dec = nn.Linear(self.z_dim, self.hidden_dim)
        self.d_rnn = RNN(self.emotion_dim + self.hidden_dim, self.hidden_dim, cell_type=self.rnn_type)
        self.d_mlp = MLP(self.hidden_dim, self.emb_dims)
        self.d_out = nn.Linear(self.d_mlp.out_dim, self.emotion_dim)
        self.d_rnn.set_mode('step')
        # decode_3dmm
        # linear layers from emotion_dim to coeff_3dmm_dim
        self.coeff_reg = nn.Sequential(
            
            nn.Linear(self.emotion_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.coeff_3dmm_dim),
        )
        
        self.dropout = nn.Dropout(self.dropout)

        self.quantize = VectorQuantizer(200,128,beta=0.25)

    def _encode(self, x):
        assert x.shape[0] == self.window_size # otherwise it does not make sense
        return self.x_rnn(x)[-1]
    
    def _decode(self, h_y):
        z_q, emb_loss, info = self.quantize(h_y)
        h_y = self.fc_z_dec(h_y)
        h_y = self.dropout(h_y)
        self.d_rnn.initialize(batch_size=h_y.shape[0])
        y = []
        for i in range(self.window_size):
            y_p = torch.zeros((h_y.shape[0], self.emotion_dim), device=h_y.device) if i == 0 else y_i
            rnn_in = torch.cat([h_y, y_p], dim=1)
            h = self.d_rnn(rnn_in)
            h = self.d_mlp(h)
            y_i = self.d_out(h)
            y.append(y_i)

        return torch.stack(y), emb_loss, info
    
    def encode_all(self, emotion_seq):
        # emotion_seq: (batch_size, seq_len, emotion_dim)
        # we encode the emotion_seq in windows of size window_size
        batch_size, seq_len = emotion_seq.shape[:2]
        assert seq_len % self.window_size == 0, "seq_len must be divisible by window_size"
        emotion_seq = emotion_seq.reshape(batch_size * (seq_len // self.window_size), self.window_size, -1)
        # as a result, we have the (seq_len // window_length) windows for each batch element, sequentially

        emotion_seq = rearrange(emotion_seq, 'b s f -> s b f' )

        # original code from DLow repository
        h_x = self._encode(emotion_seq) # [batch_size * num_windows, features_size]
        return h_x
    
    def encode(self, emotion_seq):
        # if seq_len > self.window_size --> sample a random window from emotion_seq
        # if seq_len == self.window_size --> encode the whole emotion_seq
        batch_size, seq_len = emotion_seq.shape[:2]
        
        if seq_len > self.window_size:
            selected_window = np.random.randint(0, seq_len // self.window_size)
            selected_emotion_seq = emotion_seq[:, selected_window * self.window_size : (selected_window + 1) * self.window_size, :]
        elif seq_len == self.window_size:
            selected_emotion_seq = emotion_seq
        else:
            raise ValueError("seq_len must be at least window_size")
        
        return self._encode(rearrange(selected_emotion_seq, 'b s f -> s b f' ))

    def decode(self, emb):
        Y_r = self._decode(emb)
        # BACK TO ORIGINAL SHAPE
        Y_r = rearrange(Y_r[0], "s b f -> b s f")
        return Y_r
    
    
    
    def decode_coeff(self, emotion):
        return self.coeff_reg(emotion)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, listener_emotion=None, listener_3dmm=None, **kwargs):
        """
        video: ...
        audio: ...
        speaker_emotion: (batch_size, seq_len (750), emotion_dim (25))
        """
        batch_size, frame_num = listener_emotion.shape[:2]

        # stack window_size frames together
        # from [batch_size, seq_len, emotion_dim] to [batch_size * (seq_len // window_length), window_length, emotion_dim]
        emotion = listener_emotion.reshape(batch_size * (frame_num // self.window_size), self.window_size, -1)
        # as a result, we have the (seq_len // window_length) windows for each batch element, sequentially
        # print('aaaa emotion hsape: ', emotion.shape)
        emotion = rearrange(emotion, 'b s f -> s b f' )
        # print('emotion hsape: ', emotion.shape)
        # original code from DLow repository
        h_x = self._encode(emotion)
        # mu = self.fc_mu_enc(h_x)
        # logvar = self.fc_logvar_enc(h_x)
        # if self.training:
        #     z = self.reparameterize(mu, logvar)
        # else:
        #     z = mu
        # z_q, emb_loss, info = self.quantize(h_x) ## finds nearest quantization
        #z_q, loss, (perplexity, min_encodings, min_encoding_indices)


        Y_r, emb_loss, info = self._decode(h_x)
        # BACK TO ORIGINAL SHAPE
        Y_r = rearrange(Y_r, "s b f -> b s f")
        Y_r = Y_r.reshape(batch_size, frame_num, -1) # undo the window-size stacking

        return {
            "prediction": Y_r,
            "target": listener_emotion, # we are autoencoding
            "coefficients_3dmm": self.decode_coeff(Y_r), 
            "target_coefficients": listener_3dmm, 
            "emb_loss": emb_loss, 
            "info": info, 
        }

class TransEncoder(nn.Module):
    def __init__(self,
                 in_channels: int,
                 latent_dim: int = 256,
                 **kwargs) -> None:
        super(TransEncoder, self).__init__()

        self.latent_dim = latent_dim
        self.in_channels = in_channels
        self.linear = nn.Linear(in_channels, latent_dim)

        seq_trans_encoder_layer = nn.TransformerEncoderLayer(d_model=latent_dim,
                                                             nhead=4,
                                                             dim_feedforward=latent_dim * 2,
                                                             dropout=0)

        self.seqTransEncoder = nn.TransformerEncoder(seq_trans_encoder_layer, num_layers=1)
        self.mu_token = nn.Parameter(torch.randn(latent_dim))
        self.logvar_token = nn.Parameter(torch.randn(latent_dim))


    def forward(self, input):
        input = input.permute(1,0,2)
        x = self.linear(input)  # B T D
        B, T, D = input.shape

        lengths = [len(item) for item in input]

        # mu_token = torch.tile(self.mu_token, (B,)).reshape(B, 1, -1)
        # logvar_token = torch.tile(self.logvar_token, (B,)).reshape(B, 1, -1)

        # x = torch.cat([mu_token, logvar_token, x], dim=1)

        x = x.permute(1, 0, 2)
     
        token_mask = torch.ones((B, 0), dtype=bool, device=input.get_device())
        mask = lengths_to_mask(lengths, input.get_device())

        aug_mask = torch.cat((token_mask, mask), 1)

        x = self.seqTransEncoder(x, src_key_padding_mask=~aug_mask)

        # mu = x[0]
        # logvar = x[1]
        # std = logvar.exp().pow(0.5)
        # dist = torch.distributions.Normal(mu, std)
        # motion_sample = self.sample_from_distribution(dist).to(input.get_device())
        return x
def init_biased_mask(n_head, max_seq_len, period):
    def get_slopes(n):
        def get_slopes_power_of_2(n):
            start = (2**(-2**-(math.log2(n)-3)))
            ratio = start
            return [start*ratio**i for i in range(n)]
        if math.log2(n).is_integer():
            return get_slopes_power_of_2(n)
        else:
            closest_power_of_2 = 2**math.floor(math.log2(n))
            return get_slopes_power_of_2(closest_power_of_2) + get_slopes(2*closest_power_of_2)[0::2][:n-closest_power_of_2]
    slopes = torch.Tensor(get_slopes(n_head))
    bias = torch.arange(start=0, end=max_seq_len, step=period).unsqueeze(1).repeat(1,period).view(-1)//(period)
    bias = - torch.flip(bias,dims=[0])
    alibi = torch.zeros(max_seq_len, max_seq_len)
    for i in range(max_seq_len):
        alibi[i, :i+1] = bias[-(i+1):]
    alibi = slopes.unsqueeze(1).unsqueeze(1) * alibi.unsqueeze(0)
    mask = (torch.triu(torch.ones(max_seq_len, max_seq_len)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    mask = mask.unsqueeze(0) + alibi
    return mask


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000, batch_first=True):
        super().__init__()
        self.batch_first = batch_first

        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)

        self.register_buffer('pe', pe)

    def forward(self, x):
        # not used in the final model
        if self.batch_first:
            x = x + self.pe.permute(1, 0, 2)[:, :x.shape[1], :]
        else:
            x = x + self.pe[:x.shape[0], :]
        return self.dropout(x)


class TransDecoder(nn.Module):
    def __init__(self,  output_3dmm_dim = 58, feature_dim = 128, device = 'cuda:0', max_seq_len=750, n_head = 4, window_size = 8, online = True):
        super(TransDecoder, self).__init__()

        self.feature_dim = feature_dim
        self.window_size = window_size
        self.device = device
        self.online = online

        # self.vae_model = TransformerDecoder(feature_dim, feature_dim)

        if self.online:
            self.lstm = nn.LSTM(feature_dim, feature_dim, 1 , batch_first = True)
            self.linear_3d = nn.Linear(output_3dmm_dim, feature_dim)
            self.linear_reaction = nn.Linear(feature_dim, feature_dim)
            decoder_layer_3d = nn.TransformerDecoderLayer(d_model=feature_dim, nhead=4, dim_feedforward=2*feature_dim, batch_first=True)
            self.listener_reaction_decoder_3d = nn.TransformerDecoder(decoder_layer_3d, num_layers=1)



        decoder_layer = nn.TransformerDecoderLayer(d_model=feature_dim, nhead=n_head, dim_feedforward=2*feature_dim, batch_first=True)
        self.listener_reaction_decoder_1 = nn.TransformerDecoder(decoder_layer, num_layers=1)
        self.listener_reaction_decoder_2 = nn.TransformerDecoder(decoder_layer, num_layers=1)


        self.biased_mask = init_biased_mask(n_head = n_head, max_seq_len = max_seq_len, period=max_seq_len)

        self.listener_reaction_3dmm_map_layer = nn.Linear(feature_dim, output_3dmm_dim)
        self.PE = PositionalEncoding(feature_dim)


    def forward(self, encoded_feature, past_reaction_3dmm = None, past_reaction_emotion = None):
        B, TS = encoded_feature.shape[0], encoded_feature.shape[1]
        if self.online:
            TL = self.window_size
        else:
            TL = TS
        motion_sample = encoded_feature
        time_queries = torch.zeros(B, TL, self.feature_dim, device=encoded_feature.get_device())
        time_queries = self.PE(time_queries)
        tgt_mask = self.biased_mask[:, :TL, :TL].clone().detach().cuda().repeat(B,1,1)


        listener_reaction = self.listener_reaction_decoder_1(tgt=time_queries, memory=motion_sample.unsqueeze(1), tgt_mask=tgt_mask)
        listener_reaction = self.listener_reaction_decoder_2(listener_reaction, listener_reaction, tgt_mask=tgt_mask)



        # if self.online and (past_reaction_3dmm is not None):
        #     past_reaction_3dmm = self.linear_3d(past_reaction_3dmm)
        #     past_reaction_3dmm_last = past_reaction_3dmm[:,-1]

        #     tgt_mask = self.biased_mask[:, :(TL + past_reaction_3dmm.shape[1]), :(TL + past_reaction_3dmm.shape[1])].detach().to(device=self.device).repeat(B,1,1)
        #     all_3dmm = torch.cat((past_reaction_3dmm, self.linear_reaction(listener_reaction)), dim = 1)
        #     listener_3dmm_out = self.listener_reaction_decoder_3d(all_3dmm, all_3dmm, tgt_mask=tgt_mask)
        #     frame_num = listener_3dmm_out.shape[1]
        #     listener_3dmm_out = listener_3dmm_out[:, (frame_num - TL):]

        #     listener_3dmm_out, _ = self.lstm(listener_3dmm_out, (past_reaction_3dmm_last.view(1, B, self.feature_dim).contiguous(), past_reaction_3dmm_last.view(1, B, self.feature_dim).contiguous()))


        #     listener_3dmm_out = self.listener_reaction_3dmm_map_layer(listener_3dmm_out)
        # else:
        #     listener_3dmm_out = self.listener_reaction_3dmm_map_layer(listener_reaction)
        listener_3dmm_out = self.listener_reaction_3dmm_map_layer(listener_reaction)
        # return listener_3dmm_out, dist
        return encoded_feature 

    def reset_window_size(self, window_size):
        self.window_size = window_size

def lengths_to_mask(lengths, device):
    lengths = torch.tensor(lengths, device=device)
    max_len = max(lengths)
    mask = torch.arange(max_len, device=device).expand(len(lengths), max_len) < lengths.unsqueeze(1)
    return mask

class AutoencoderRNN_FSQ_VAE_v2(nn.Module):
    def __init__(self, cfg):
        super(AutoencoderRNN_FSQ_VAE_v2, self).__init__()
        self.seq_len = cfg.seq_len
        self.window_size = cfg.window_size
        assert self.seq_len % self.window_size == 0, "seq_len must be divisible by window_size"
        self.hidden_dim = cfg.hidden_dim
        self.z_dim = cfg.z_dim
        self.emb_dims = cfg.emb_dims
        self.num_layers = cfg.num_layers
        self.rnn_type =  cfg.rnn_type
        self.dropout = cfg.dropout
        self.emotion_dim = cfg.emotion_dim
        self.coeff_3dmm_dim = cfg.coeff_3dmm_dim

        # encode
        self.x_rnn = TransEncoder(25,128)
        # z
        # self.fc_mu_enc = nn.Linear(self.hidden_dim, self.z_dim)
        # self.fc_logvar_enc = nn.Linear(self.hidden_dim, self.z_dim)
        
        # decode
        self.fc_z_dec = nn.Linear(self.z_dim, self.hidden_dim)
        self.d_rnn = TransDecoder(output_3dmm_dim = 128, feature_dim = 128,  device = 'cuda:1', max_seq_len=751, n_head = 4, window_size = 10, online = False)
        self.d_mlp = MLP(self.hidden_dim, self.emb_dims)
        self.d_out = nn.Linear(self.d_mlp.out_dim, self.emotion_dim)
        # self.d_rnn.set_mode('step')
        # decode_3dmm
        # linear layers from emotion_dim to coeff_3dmm_dim
        self.coeff_reg = nn.Sequential(
            
            nn.Linear(self.emotion_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.coeff_3dmm_dim),
        )
        
        self.dropout = nn.Dropout(self.dropout)

        self.quantize = VectorQuantizer(200,128,beta=0.25)

    def _encode(self, x):
        assert x.shape[0] == self.window_size # otherwise it does not make sense
        # print('rrn in', self.x_rnn(x)[-1].shape)
        return self.x_rnn(x)[-1]
    
    def _decode(self, h_y):
        # z_q, emb_loss, info = self.quantize(h_y)
        h_y = self.fc_z_dec(h_y)
        h_y = self.dropout(h_y)
        # self.d_rnn.initialize(batch_size=h_y.shape[0])
        y = []
        for i in range(self.window_size):
            # y_p = torch.zeros((h_y.shape[0], self.emotion_dim), device=h_y.device) if i == 0 else y_i
            # rnn_in = torch.cat([h_y, y_p], dim=1)
            # print('rrn out', h_y.shape)
            h = self.d_rnn(h_y)
            # print('rrn out', h.shape)
            h = self.d_mlp(h)
            y_i = self.d_out(h)
            y.append(y_i)
        emb_loss, info = 0, None
        return torch.stack(y), emb_loss, info
    
    def encode_all(self, emotion_seq):
        # emotion_seq: (batch_size, seq_len, emotion_dim)
        # we encode the emotion_seq in windows of size window_size
        batch_size, seq_len = emotion_seq.shape[:2]
        assert seq_len % self.window_size == 0, "seq_len must be divisible by window_size"
        emotion_seq = emotion_seq.reshape(batch_size * (seq_len // self.window_size), self.window_size, -1)
        # as a result, we have the (seq_len // window_length) windows for each batch element, sequentially

        emotion_seq = rearrange(emotion_seq, 'b s f -> s b f' )

        # original code from DLow repository
        h_x = self._encode(emotion_seq) # [batch_size * num_windows, features_size]
        return h_x
    
    def encode(self, emotion_seq):
        # if seq_len > self.window_size --> sample a random window from emotion_seq
        # if seq_len == self.window_size --> encode the whole emotion_seq
        batch_size, seq_len = emotion_seq.shape[:2]
        
        if seq_len > self.window_size:
            selected_window = np.random.randint(0, seq_len // self.window_size)
            selected_emotion_seq = emotion_seq[:, selected_window * self.window_size : (selected_window + 1) * self.window_size, :]
        elif seq_len == self.window_size:
            selected_emotion_seq = emotion_seq
        else:
            raise ValueError("seq_len must be at least window_size")
        
        return self._encode(rearrange(selected_emotion_seq, 'b s f -> s b f' ))

    def decode(self, emb):
        Y_r = self._decode(emb)
        # BACK TO ORIGINAL SHAPE
        Y_r = rearrange(Y_r[0], "s b f -> b s f")
        return Y_r
    
    
    
    def decode_coeff(self, emotion):
        return self.coeff_reg(emotion)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, listener_emotion=None, listener_3dmm=None, **kwargs):
        """
        video: ...
        audio: ...
        speaker_emotion: (batch_size, seq_len (750), emotion_dim (25))
        """
        batch_size, frame_num = listener_emotion.shape[:2]

        # stack window_size frames together
        # from [batch_size, seq_len, emotion_dim] to [batch_size * (seq_len // window_length), window_length, emotion_dim]
        emotion = listener_emotion.reshape(batch_size * (frame_num // self.window_size), self.window_size, -1)
        # as a result, we have the (seq_len // window_length) windows for each batch element, sequentially
        # print('aaaa emotion hsape: ', emotion.shape)
        emotion = rearrange(emotion, 'b s f -> s b f' )

        # original code from DLow repository
        h_x = self._encode(emotion)
        # mu = self.fc_mu_enc(h_x)
        # logvar = self.fc_logvar_enc(h_x)
        # if self.training:
        #     z = self.reparameterize(mu, logvar)
        # else:
        #     z = mu
        # z_q, emb_loss, info = self.quantize(h_x) ## finds nearest quantization
        #z_q, loss, (perplexity, min_encodings, min_encoding_indices)


        Y_r, emb_loss, info = self._decode(h_x)
        # BACK TO ORIGINAL SHAPE
        Y_r = rearrange(Y_r, "s b f -> b s f")
        Y_r = Y_r.reshape(batch_size, frame_num, -1) # undo the window-size stacking

        return {
            "prediction": Y_r,
            "target": listener_emotion, # we are autoencoding
            "coefficients_3dmm": self.decode_coeff(Y_r), 
            "target_coefficients": listener_3dmm, 
            "emb_loss": emb_loss, 
            "info": info, 
        }


class VectorQuantizer(nn.Module):
    """
    see https://github.com/MishaLaskin/vqvae/blob/d761a999e2267766400dc646d82d3ac3657771d4/models/quantizer.py
    ____________________________________________
    Discretization bottleneck part of the VQ-VAE.
    Inputs:
    - n_e : number of embeddings
    - e_dim : dimension of embedding
    - beta : commitment cost used in loss term, beta * ||z_e(x)-sg[e]||^2
    _____________________________________________
    """

    def __init__(self, n_e, e_dim, beta):
        super(VectorQuantizer, self).__init__()
        self.n_e = n_e
        self.e_dim = e_dim
        self.beta = beta

        self.embedding = nn.Embedding(self.n_e, self.e_dim)
        self.embedding.weight.data.uniform_(-1.0 / self.n_e, 1.0 / self.n_e)

    def forward(self, z):
        """
        Inputs the output of the encoder network z and maps it to a discrete
        one-hot vector that is the index of the closest embedding vector e_j
        z (continuous) -> z_q (discrete)
        z.shape = (batch, channel, height, width)
        quantization pipeline:
            1. get encoder input (B,C,H,W)
            2. flatten input to (B*H*W,C)
        """
        # reshape z -> (batch, height, width, channel) and flatten
        #print('zshape', z.shape)
        # z = z.permute(0, 2, 1).contiguous()
        # z_flattened = z.view(-1, self.e_dim)
        z_flattened = z
        # distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z

        d = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + \
            torch.sum(self.embedding.weight**2, dim=1) - 2 * \
            torch.matmul(z_flattened, self.embedding.weight.t())

        ## could possible replace this here
        # #\start...
        # find closest encodings
        min_encoding_indices = torch.argmin(d, dim=1).unsqueeze(1)

        min_encodings = torch.zeros(
            min_encoding_indices.shape[0], self.n_e).to(z)
        min_encodings.scatter_(1, min_encoding_indices, 1)

        # dtype min encodings: torch.float32
        # min_encodings shape: torch.Size([2048, 512])
        # min_encoding_indices.shape: torch.Size([2048, 1])

        # get quantized latent vectors
        z_q = torch.matmul(min_encodings, self.embedding.weight).view(z.shape)
        #.........\end

        # with:
        # .........\start
        #min_encoding_indices = torch.argmin(d, dim=1)
        #z_q = self.embedding(min_encoding_indices)
        # ......\end......... (TODO)

        # compute loss for embedding
        loss = self.beta * torch.mean((z_q.detach()-z)**2) + \
                   torch.mean((z_q - z.detach()) ** 2)
        #loss = torch.mean((z_q.detach()-z)**2) + self.beta * \
        #    torch.mean((z_q - z.detach()) ** 2)

        # preserve gradients
        z_q = z + (z_q - z).detach()

        # perplexity
        e_mean = torch.mean(min_encodings, dim=0)
        perplexity = torch.exp(-torch.sum(e_mean * torch.log(e_mean + 1e-10)))

        # # reshape back to match original input shape
        # z_q = z_q.permute(0, 2, 1).contiguous()

        return z_q, loss, (perplexity, min_encodings, min_encoding_indices)

    def get_distance(self, z):
        z = z.permute(0, 2, 1).contiguous()
        z_flattened = z.view(-1, self.e_dim)
        # distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z

        d = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + \
            torch.sum(self.embedding.weight**2, dim=1) - 2 * \
            torch.matmul(z_flattened, self.embedding.weight.t())
        d = torch.reshape(d, (z.shape[0], -1, z.shape[2])).permute(0,2,1).contiguous()
        return d

    def get_codebook_entry(self, indices, shape):
        # shape specifying (batch, height, width, channel)
        # TODO: check for more easy handling with nn.Embedding
        min_encodings = torch.zeros(indices.shape[0], self.n_e).to(indices)
        min_encodings.scatter_(1, indices[:,None], 1)

        # get quantized latent vectors
        #print(min_encodings.shape, self.embedding.weight.shape)
        z_q = torch.matmul(min_encodings.float(), self.embedding.weight)

        if shape is not None:
            z_q = z_q.view(shape)

            # reshape back to match original input shape
            #z_q = z_q.permute(0, 3, 1, 2).contiguous()

        return z_q