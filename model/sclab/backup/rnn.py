
import torch
import torch.nn as nn
from model.belfusion.torch import *
from einops import rearrange

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

        self.quantize = VectorQuantizer(256,512,beta=0)

        for param in self.x_rnn.parameters():
            param.requires_grad = False
        for param in self.fc_z_dec.parameters():
            param.requires_grad = False
        for param in self.d_rnn.parameters():
            param.requires_grad = False
        for param in self.d_mlp.parameters():
            param.requires_grad = False
        for param in self.d_out.parameters():
            param.requires_grad = False
        for param in self.dropout.parameters():
            param.requires_grad = False
        for param in self.quantize.parameters():
            param.requires_grad = False       

    def _encode(self, x):
        assert x.shape[0] == self.window_size # otherwise it does not make sense
        return self.x_rnn(x)[-1]
    
    def _decode(self, h_y):
        z_q, emb_loss, info = self.quantize(h_y)
        h_y = self.fc_z_dec(z_q)
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