# forecast/models/MTGNN.py
import torch
import torch.nn as nn
from torch.nn import init
import numbers
import torch.nn.functional as F


# --- layer.py content merged and adapted ---

class nconv(nn.Module):
    """
    Graph convolution operation.
    Performs element-wise multiplication of features with adjacency matrix.
    """

    def __init__(self):
        super(nconv, self).__init__()

    def forward(self, x, A):
        # x: (batch_size, num_channels, num_nodes, sequence_length)
        # A: (num_nodes, num_nodes)
        # Result: (batch_size, num_channels, num_nodes, sequence_length)
        x = torch.einsum('ncwl,vw->ncvl', (x, A))
        return x.contiguous()


class dy_nconv(nn.Module):
    """
    Dynamic graph convolution operation.
    Performs element-wise multiplication of features with dynamic adjacency matrix.
    """

    def __init__(self):
        super(dy_nconv, self).__init__()

    def forward(self, x, A):
        # x: (batch_size, num_channels, num_nodes, sequence_length)
        # A: (batch_size, num_nodes, num_nodes) - dynamic adjacency
        # Result: (batch_size, num_channels, num_nodes, sequence_length)
        x = torch.einsum('ncvl,nvwl->ncwl', (x, A))
        return x.contiguous()


class linear(nn.Module):
    """
    1x1 Convolution for feature transformation.
    """

    def __init__(self, c_in, c_out, bias=True):
        super(linear, self).__init__()
        self.mlp = torch.nn.Conv2d(c_in, c_out, kernel_size=(1, 1), padding=(0, 0), stride=(1, 1), bias=bias)

    def forward(self, x):
        return self.mlp(x)


class prop(nn.Module):
    """
    Propagation layer for static graph convolution.
    """

    def __init__(self, c_in, c_out, gdep, dropout, alpha):
        super(prop, self).__init__()
        self.nconv = nconv()
        self.mlp = linear(c_in, c_out)
        self.gdep = gdep
        self.dropout = dropout
        self.alpha = alpha

    def forward(self, x, adj):
        adj = adj + torch.eye(adj.size(0)).to(x.device)  # Add self-loops
        d = adj.sum(1)  # Degree matrix
        h = x
        # Normalize adjacency matrix
        a = adj / d.view(-1, 1)
        for i in range(self.gdep):
            h = self.alpha * x + (1 - self.alpha) * self.nconv(h, a)
        ho = self.mlp(h)
        return ho


class mixprop(nn.Module):
    """
    Mixed propagation layer for static graph convolution.
    Concatenates intermediate propagation results.
    """

    def __init__(self, c_in, c_out, gdep, dropout, alpha):
        super(mixprop, self).__init__()
        self.nconv = nconv()
        self.mlp = linear((gdep + 1) * c_in, c_out)
        self.gdep = gdep
        self.dropout = dropout
        self.alpha = alpha

    def forward(self, x, adj):
        adj = adj + torch.eye(adj.size(0)).to(x.device)  # Add self-loops
        d = adj.sum(1)  # Degree matrix
        h = x
        out = [h]
        # Normalize adjacency matrix
        a = adj / d.view(-1, 1)
        for i in range(self.gdep):
            h = self.alpha * x + (1 - self.alpha) * self.nconv(h, a)
            out.append(h)
        ho = torch.cat(out, dim=1)  # Concatenate all intermediate results
        ho = self.mlp(ho)
        return ho


class dy_mixprop(nn.Module):
    """
    Dynamic mixed propagation layer.
    Constructs dynamic adjacency matrices based on input features.
    """

    def __init__(self, c_in, c_out, gdep, dropout, alpha):
        super(dy_mixprop, self).__init__()
        self.nconv = dy_nconv()
        self.mlp1 = linear((gdep + 1) * c_in, c_out)
        self.mlp2 = linear((gdep + 1) * c_in, c_out)

        self.gdep = gdep
        self.dropout = dropout
        self.alpha = alpha
        self.lin1 = linear(c_in, c_in)
        self.lin2 = linear(c_in, c_in)

    def forward(self, x):
        # Generate dynamic adjacency matrices
        x1 = torch.tanh(self.lin1(x))
        x2 = torch.tanh(self.lin2(x))
        adj = self.nconv(x1.transpose(2, 1), x2)
        adj0 = torch.softmax(adj, dim=2)  # Row-wise softmax
        adj1 = torch.softmax(adj.transpose(2, 1), dim=2)  # Column-wise softmax

        # Propagate with adj0
        h = x
        out = [h]
        for i in range(self.gdep):
            h = self.alpha * x + (1 - self.alpha) * self.nconv(h, adj0)
            out.append(h)
        ho = torch.cat(out, dim=1)
        ho1 = self.mlp1(ho)

        # Propagate with adj1
        h = x
        out = [h]
        for i in range(self.gdep):
            h = self.alpha * x + (1 - self.alpha) * self.nconv(h, adj1)
            out.append(h)
        ho = torch.cat(out, dim=1)
        ho2 = self.mlp2(ho)

        return ho1 + ho2


class dilated_1D(nn.Module):
    """
    Dilated 1D Convolution.
    """

    def __init__(self, cin, cout, dilation_factor=2):
        super(dilated_1D, self).__init__()
        self.tconv = nn.Conv2d(cin, cout, (1, 7), dilation=(1, dilation_factor))

    def forward(self, input):
        x = self.tconv(input)
        return x


class dilated_inception(nn.Module):
    """
    Dilated Inception block with multiple kernel sizes.
    """

    def __init__(self, cin, cout, dilation_factor=2):
        super(dilated_inception, self).__init__()
        self.tconv = nn.ModuleList()
        self.kernel_set = [2, 3, 6, 7]
        cout = int(cout / len(self.kernel_set))
        for kern in self.kernel_set:
            self.tconv.append(nn.Conv2d(cin, cout, (1, kern), dilation=(1, dilation_factor)))

    def forward(self, input):
        x = []
        for i in range(len(self.kernel_set)):
            x.append(self.tconv[i](input))
        # Pad to match the output length of the largest kernel
        for i in range(len(self.kernel_set)):
            x[i] = x[i][..., -x[-1].size(3):]
        x = torch.cat(x, dim=1)
        return x


class graph_constructor(nn.Module):
    """
    Constructs a graph adjacency matrix based on node embeddings.
    Supports static features or learned embeddings.
    """

    def __init__(self, nnodes, k, dim, device, alpha=3, static_feat=None):
        super(graph_constructor, self).__init__()
        self.nnodes = nnodes
        if static_feat is not None:
            xd = static_feat.shape[1]
            self.lin1 = nn.Linear(xd, dim)
            self.lin2 = nn.Linear(xd, dim)
        else:
            self.emb1 = nn.Embedding(nnodes, dim)
            self.emb2 = nn.Embedding(nnodes, dim)
            self.lin1 = nn.Linear(dim, dim)
            self.lin2 = nn.Linear(dim, dim)

        self.device = device
        self.k = k
        self.dim = dim
        self.alpha = alpha
        self.static_feat = static_feat

    def forward(self, idx):
        if self.static_feat is None:
            nodevec1 = self.emb1(idx)
            nodevec2 = self.emb2(idx)
        else:
            nodevec1 = self.static_feat[idx, :]
            nodevec2 = nodevec1

        nodevec1 = torch.tanh(self.alpha * self.lin1(nodevec1))
        nodevec2 = torch.tanh(self.alpha * self.lin2(nodevec2))

        # Adjacency matrix calculation (based on original MTGNN paper)
        a = torch.mm(nodevec1, nodevec2.transpose(1, 0)) - torch.mm(nodevec2, nodevec1.transpose(1, 0))
        adj = F.relu(torch.tanh(self.alpha * a))

        # Apply top-k mask to create sparse graph
        mask = torch.zeros(idx.size(0), idx.size(0)).to(self.device)
        mask.fill_(float('0'))
        s1, t1 = (adj + torch.rand_like(adj) * 0.01).topk(self.k, 1)
        mask.scatter_(1, t1, s1.fill_(1))
        adj = adj * mask
        return adj

    def fullA(self, idx):
        """
        Returns the full (unmasked) adjacency matrix.
        """
        if self.static_feat is None:
            nodevec1 = self.emb1(idx)
            nodevec2 = self.emb2(idx)
        else:
            nodevec1 = self.static_feat[idx, :]
            nodevec2 = nodevec1

        nodevec1 = torch.tanh(self.alpha * self.lin1(nodevec1))
        nodevec2 = torch.tanh(self.alpha * self.lin2(nodevec2))

        a = torch.mm(nodevec1, nodevec2.transpose(1, 0)) - torch.mm(nodevec2, nodevec1.transpose(1, 0))
        adj = F.relu(torch.tanh(self.alpha * a))
        return adj


class graph_global(nn.Module):
    """
    Learns a global, static adjacency matrix.
    """

    def __init__(self, nnodes, k, dim, device, alpha=3, static_feat=None):
        super(graph_global, self).__init__()
        self.nnodes = nnodes
        self.A = nn.Parameter(torch.randn(nnodes, nnodes).to(device), requires_grad=True).to(device)

    def forward(self, idx):
        return F.relu(self.A)


class graph_undirected(nn.Module):
    """
    Constructs an undirected graph adjacency matrix.
    """

    def __init__(self, nnodes, k, dim, device, alpha=3, static_feat=None):
        super(graph_undirected, self).__init__()
        self.nnodes = nnodes
        if static_feat is not None:
            xd = static_feat.shape[1]
            self.lin1 = nn.Linear(xd, dim)
        else:
            self.emb1 = nn.Embedding(nnodes, dim)
            self.lin1 = nn.Linear(dim, dim)

        self.device = device
        self.k = k
        self.dim = dim
        self.alpha = alpha
        self.static_feat = static_feat

    def forward(self, idx):
        if self.static_feat is None:
            nodevec1 = self.emb1(idx)
            nodevec2 = self.emb1(idx)
        else:
            nodevec1 = self.static_feat[idx, :]
            nodevec2 = nodevec1

        nodevec1 = torch.tanh(self.alpha * self.lin1(nodevec1))
        nodevec2 = torch.tanh(self.alpha * self.lin1(nodevec2))

        a = torch.mm(nodevec1, nodevec2.transpose(1, 0))
        adj = F.relu(torch.tanh(self.alpha * a))
        mask = torch.zeros(idx.size(0), idx.size(0)).to(self.device)
        mask.fill_(float('0'))
        s1, t1 = adj.topk(self.k, 1)
        mask.scatter_(1, t1, s1.fill_(1))
        adj = adj * mask
        return adj


class graph_directed(nn.Module):
    """
    Constructs a directed graph adjacency matrix.
    """

    def __init__(self, nnodes, k, dim, device, alpha=3, static_feat=None):
        super(graph_directed, self).__init__()
        self.nnodes = nnodes
        if static_feat is not None:
            xd = static_feat.shape[1]
            self.lin1 = nn.Linear(xd, dim)
            self.lin2 = nn.Linear(xd, dim)
        else:
            self.emb1 = nn.Embedding(nnodes, dim)
            self.emb2 = nn.Embedding(nnodes, dim)
            self.lin1 = nn.Linear(dim, dim)
            self.lin2 = nn.Linear(dim, dim)

        self.device = device
        self.k = k
        self.dim = dim
        self.alpha = alpha
        self.static_feat = static_feat

    def forward(self, idx):
        if self.static_feat is None:
            nodevec1 = self.emb1(idx)
            nodevec2 = self.emb2(idx)
        else:
            nodevec1 = self.static_feat[idx, :]
            nodevec2 = nodevec1

        nodevec1 = torch.tanh(self.alpha * self.lin1(nodevec1))
        nodevec2 = torch.tanh(self.alpha * self.lin2(nodevec2))

        a = torch.mm(nodevec1, nodevec2.transpose(1, 0))
        adj = F.relu(torch.tanh(self.alpha * a))
        mask = torch.zeros(idx.size(0), idx.size(0)).to(self.device)
        mask.fill_(float('0'))
        s1, t1 = adj.topk(self.k, 1)
        mask.scatter_(1, t1, s1.fill_(1))
        adj = adj * mask
        return adj


class LayerNorm(nn.Module):
    """
    Custom Layer Normalization module.
    """
    __constants__ = ['normalized_shape', 'weight', 'bias', 'eps', 'elementwise_affine']

    def __init__(self, normalized_shape, eps=1e-5, elementwise_affine=True):
        super(LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = tuple(normalized_shape)
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.Tensor(*normalized_shape))
            self.bias = nn.Parameter(torch.Tensor(*normalized_shape))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        if self.elementwise_affine:
            init.ones_(self.weight)
            init.zeros_(self.bias)

    def forward(self, input, idx):
        # Apply layer normalization.
        # Note: Original MTGNN LayerNorm takes an `idx` argument which is used for slicing weights/biases.
        # This is unusual for standard LayerNorm and might indicate a specific graph-related normalization.
        # For general integration, we might need to simplify or ensure `idx` is correctly handled.
        if self.elementwise_affine:
            # Assuming idx is used to select specific normalization parameters per node/feature
            # This might need adjustment based on how `input` and `idx` are used in practice.
            # The original code uses input.shape[1:] for normalized_shape, which means it normalizes
            # over the channels, nodes, and sequence length dimensions.
            return F.layer_norm(input, tuple(input.shape[1:]), self.weight[:, idx, :], self.bias[:, idx, :], self.eps)
        else:
            return F.layer_norm(input, tuple(input.shape[1:]), self.weight, self.bias, self.eps)

    def extra_repr(self):
        return '{normalized_shape}, eps={eps}, ' \
               'elementwise_affine={elementwise_affine}'.format(**self.__dict__)


# --- net.py content merged and adapted, gtnet renamed to Model ---

class Model(nn.Module):  # Renamed gtnet to Model for consistency with forecast framework
    def __init__(self, configs):  # configs object will contain all necessary arguments
        super(Model, self).__init__()

        # Map configs arguments to MTGNN's gtnet parameters
        self.gcn_true = getattr(configs, 'gcn_true', True)
        self.buildA_true = getattr(configs, 'buildA_true', True)
        self.gcn_depth = getattr(configs, 'gcn_depth', 2)
        self.num_nodes = configs.enc_in  # Assuming enc_in is the number of nodes (N)
        self.device = torch.device(f'cuda:{configs.gpu}' if torch.cuda.is_available() and configs.use_gpu else 'cpu')
        self.dropout = configs.dropout
        self.subgraph_size = getattr(configs, 'subgraph_size', 20)
        self.node_dim = getattr(configs, 'node_dim', 40)
        self.dilation_exponential = getattr(configs, 'dilation_exponential', 1)
        self.conv_channels = getattr(configs, 'conv_channels', 32)
        self.residual_channels = getattr(configs, 'residual_channels', 32)
        self.skip_channels = getattr(configs, 'skip_channels', 64)
        self.end_channels = getattr(configs, 'end_channels', 128)
        self.seq_length = configs.seq_len  # Input sequence length
        self.in_dim = 1  # Assuming single feature per node for now, adjust if 'M' features means multi-feature input
        self.out_dim = configs.pred_len  # Output sequence length (horizon)
        self.layers = getattr(configs, 'mtgnn_layers', 3)  # Renamed to avoid conflict with e_layers/d_layers
        self.propalpha = getattr(configs, 'propalpha', 0.05)
        self.tanhalpha = getattr(configs, 'tanhalpha', 3)
        self.layer_norm_affline = getattr(configs, 'layer_norm_affline', True)  # Original default was True

        # Predefined adjacency matrix (if applicable, currently not passed from run.py)
        self.predefined_A = None  # For now, assume adaptive graph construction

        # Static features (if applicable, currently not passed from run.py)
        self.static_feat = None

        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        self.residual_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        self.gconv1 = nn.ModuleList()
        self.gconv2 = nn.ModuleList()
        self.norm = nn.ModuleList()

        self.start_conv = nn.Conv2d(in_channels=self.in_dim,
                                    out_channels=self.residual_channels,
                                    kernel_size=(1, 1))

        self.gc = graph_constructor(self.num_nodes, self.subgraph_size, self.node_dim, self.device,
                                    alpha=self.tanhalpha, static_feat=self.static_feat)

        kernel_size = 7
        if self.dilation_exponential > 1:
            self.receptive_field = int(1 + (kernel_size - 1) * (self.dilation_exponential ** self.layers - 1) / (
                        self.dilation_exponential - 1))
        else:
            self.receptive_field = self.layers * (kernel_size - 1) + 1

        for i in range(1):  # Original MTGNN code iterates once here for the first block
            if self.dilation_exponential > 1:
                rf_size_i = int(1 + i * (kernel_size - 1) * (self.dilation_exponential ** self.layers - 1) / (
                            self.dilation_exponential - 1))
            else:
                rf_size_i = i * self.layers * (kernel_size - 1) + 1
            new_dilation = 1
            for j in range(1, self.layers + 1):
                if self.dilation_exponential > 1:
                    rf_size_j = int(rf_size_i + (kernel_size - 1) * (self.dilation_exponential ** j - 1) / (
                                self.dilation_exponential - 1))
                else:
                    rf_size_j = rf_size_i + j * (kernel_size - 1)

                self.filter_convs.append(
                    dilated_inception(self.residual_channels, self.conv_channels, dilation_factor=new_dilation))
                self.gate_convs.append(
                    dilated_inception(self.residual_channels, self.conv_channels, dilation_factor=new_dilation))
                self.residual_convs.append(nn.Conv2d(in_channels=self.conv_channels,
                                                     out_channels=self.residual_channels,
                                                     kernel_size=(1, 1)))
                if self.seq_length > self.receptive_field:
                    self.skip_convs.append(nn.Conv2d(in_channels=self.conv_channels,
                                                     out_channels=self.skip_channels,
                                                     kernel_size=(1, self.seq_length - rf_size_j + 1)))
                else:
                    self.skip_convs.append(nn.Conv2d(in_channels=self.conv_channels,
                                                     out_channels=self.skip_channels,
                                                     kernel_size=(1, self.receptive_field - rf_size_j + 1)))

                if self.gcn_true:
                    self.gconv1.append(mixprop(self.conv_channels, self.residual_channels, self.gcn_depth, self.dropout,
                                               self.propalpha))
                    self.gconv2.append(mixprop(self.conv_channels, self.residual_channels, self.gcn_depth, self.dropout,
                                               self.propalpha))

                if self.seq_length > self.receptive_field:
                    self.norm.append(
                        LayerNorm((self.residual_channels, self.num_nodes, self.seq_length - rf_size_j + 1),
                                  elementwise_affine=self.layer_norm_affline))
                else:
                    self.norm.append(
                        LayerNorm((self.residual_channels, self.num_nodes, self.receptive_field - rf_size_j + 1),
                                  elementwise_affine=self.layer_norm_affline))

                new_dilation *= self.dilation_exponential

        self.layers = self.layers  # Re-assign after loop to ensure correct value
        self.end_conv_1 = nn.Conv2d(in_channels=self.skip_channels,
                                    out_channels=self.end_channels,
                                    kernel_size=(1, 1),
                                    bias=True)
        self.end_conv_2 = nn.Conv2d(in_channels=self.end_channels,
                                    out_channels=self.out_dim,  # out_dim is pred_len
                                    kernel_size=(1, 1),
                                    bias=True)
        if self.seq_length > self.receptive_field:
            self.skip0 = nn.Conv2d(in_channels=self.in_dim, out_channels=self.skip_channels,
                                   kernel_size=(1, self.seq_length), bias=True)
            self.skipE = nn.Conv2d(in_channels=self.residual_channels, out_channels=self.skip_channels,
                                   kernel_size=(1, self.seq_length - self.receptive_field + 1), bias=True)

        else:
            self.skip0 = nn.Conv2d(in_channels=self.in_dim, out_channels=self.skip_channels,
                                   kernel_size=(1, self.receptive_field), bias=True)
            self.skipE = nn.Conv2d(in_channels=self.residual_channels, out_channels=self.skip_channels,
                                   kernel_size=(1, 1), bias=True)

        self.idx = torch.arange(self.num_nodes).to(self.device)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        # x_enc: (batch_size, seq_len, enc_in)
        # MTGNN's gtnet expects input: (batch_size, in_dim, num_nodes, seq_len)
        # Assuming x_enc is (B, T, N) where N is enc_in and in_dim is 1
        # Need to reshape x_enc to (B, 1, N, T)

        # Ensure x_enc has 3 dimensions (Batch, Seq_len, Num_nodes) or 4 dimensions (Batch, Seq_len, Num_nodes, Features)
        if x_enc.dim() == 3:
            # (B, T, N) -> (B, 1, N, T)
            input_gtnet = x_enc.unsqueeze(1).permute(0, 1, 3, 2)
        elif x_enc.dim() == 4:
            # (B, T, N, D) -> (B, D, N, T)
            # If D > 1, need to adjust in_dim in MTGNN __init__
            input_gtnet = x_enc.permute(0, 3, 2, 1)
            # Update in_dim if it's not 1 and data has multiple features per node
            if self.in_dim != x_enc.shape[-1]:
                self.in_dim = x_enc.shape[-1]
                # Reinitialize start_conv if in_dim changes dynamically (less ideal, better to set correctly at init)
                self.start_conv = nn.Conv2d(in_channels=self.in_dim,
                                            out_channels=self.residual_channels,
                                            kernel_size=(1, 1)).to(self.device)
                self.skip0 = nn.Conv2d(in_channels=self.in_dim, out_channels=self.skip_channels,
                                       kernel_size=(1, self.seq_length), bias=True).to(self.device)
        else:
            raise ValueError(f"Unsupported x_enc dimension: {x_enc.dim()}. Expected 3 or 4 dimensions.")

        seq_len_input = input_gtnet.size(3)
        assert seq_len_input == self.seq_length, f'Input sequence length ({seq_len_input}) not equal to preset sequence length ({self.seq_length})'

        if self.seq_length < self.receptive_field:
            input_gtnet = nn.functional.pad(input_gtnet, (self.receptive_field - self.seq_length, 0, 0, 0))

        adp = None
        if self.gcn_true:
            if self.buildA_true:
                adp = self.gc(self.idx)  # Use self.idx for graph construction
            else:
                # If predefined_A is expected, it should be passed during model initialization
                # For now, if buildA_true is False and predefined_A is None, this will cause an error.
                # Consider adding a default or raising a more specific error.
                if self.predefined_A is None:
                    raise ValueError("predefined_A must be provided if buildA_true is False.")
                adp = self.predefined_A.to(self.device)  # Ensure predefined_A is on correct device

        x = self.start_conv(input_gtnet)
        skip = self.skip0(F.dropout(input_gtnet, self.dropout, training=self.training))

        for i in range(self.layers):
            residual = x
            filter = self.filter_convs[i](x)
            filter = torch.tanh(filter)
            gate = self.gate_convs[i](x)
            gate = torch.sigmoid(gate)
            x = filter * gate
            x = F.dropout(x, self.dropout, training=self.training)
            s = x
            s = self.skip_convs[i](s)
            skip = s + skip
            if self.gcn_true:
                # Ensure adp is not None here
                if adp is None:
                    raise RuntimeError("Adjacency matrix 'adp' is None when gcn_true is True.")
                x = self.gconv1[i](x, adp) + self.gconv2[i](x, adp.transpose(1, 0))
            else:
                x = self.residual_convs[i](x)

            x = x + residual[:, :, :, -x.size(3):]
            x = self.norm[i](x, self.idx)  # Use self.idx for layer norm

        skip = self.skipE(x) + skip
        x = F.relu(skip)
        x = F.relu(self.end_conv_1(x))
        x = self.end_conv_2(x)  # Output shape: (B, pred_len, N, 1)

        # Reshape output to match (batch_size, pred_len, num_nodes * output_features_per_node)
        # Assuming output_features_per_node is 1 (as c_out in run.py is 1)
        # x: (B, pred_len, N, 1) -> (B, pred_len, N)
        output = x.squeeze(3)  # Remove the last dimension if it's 1
        return output
