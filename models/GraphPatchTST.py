import torch
from torch import nn
import torch.nn.functional as F
from layers.Transformer_EncDec import EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import PatchEmbedding
from einops import repeat
from einops import rearrange
import numpy as np

# 设置torch的打印选项，以便更好地显示张量
torch.set_printoptions(profile="short", linewidth=200)


class Flatten_Head(nn.Module):
    """
    最后的预测器
    """

    def __init__(self, n_vars, nf, target_window, head_dropout=0):
        super().__init__()
        self.n_vars = n_vars
        self.flatten = nn.Flatten(start_dim=-2)  # 将最后两维展平
        self.linear = nn.Linear(nf, target_window)  # 线性变换
        self.dropout = nn.Dropout(head_dropout)  # Dropout层

    def forward(self, x):  # x: [bs x nvars x d_model x patch_num]
        x = self.flatten(x)  # 展平操作
        x = self.linear(x)  # 线性变换
        x = self.dropout(x)  # Dropout操作
        return x


class nconv(nn.Module):
    """图卷积层"""

    def __init__(self):
        super(nconv, self).__init__()

    def forward(self, x, A):
        # 使用爱因斯坦求和约定进行矩阵乘法
        x = torch.einsum("nwl,vw->nvl", (x, A))
        return x.contiguous()  # 确保返回的张量在内存中是连续的


class mixprop(nn.Module):
    """图卷积网络层"""

    def __init__(self, c_in, c_out, gdep, dropout=0.2, alpha=0.1):
        super(mixprop, self).__init__()
        self.nconv = nconv()  # 图卷积层
        self.mlp = nn.Linear((gdep + 1) * c_in, c_out)  # 多层感知器
        self.gdep = gdep  # 图卷积层数
        self.dropout = dropout  # Dropout比率
        self.alpha = alpha  # 混合比例

    def forward(self, x, adj):
        # 计算每个节点的度数，即每个节点有多少邻居
        d = adj.sum(1)
        # 归一化邻接矩阵，每个节点的邻居连接权重除以该节点的度数
        a = adj / d.view(-1, 1)
        # 初始化节点特征矩阵h为输入特征矩阵x
        h = x
        # 初始化输出列表，包含初始特征矩阵h
        out = [h]
        # 进行多次图卷积操作，次数由self.gdep决定
        for _ in range(self.gdep):
            # h = self.alpha*x + (1-self.alpha)*self.nconv(h,a)
            h = F.dropout(h, self.dropout)
            h = self.nconv(h, a)
            out.append(h)
        ho = torch.cat(out, dim=2)
        ho = self.mlp(ho)
        return ho


class graph_constructor(nn.Module):
    """图生成器"""

    def __init__(self, nnodes, k, dim, alpha=1, static_feat=None):
        # 初始化父类，确保当前类的初始化能够继承父类的初始化方法
        super(graph_constructor, self).__init__()
        # 节点数量，表示图中节点的总数
        self.nnodes = nnodes

        # 定义两个线性层
        self.lin1 = nn.Linear(dim, dim)
        self.lin2 = nn.Linear(dim, dim)

        self.k = k
        self.dim = dim
        self.alpha = alpha
        self.static_feat = static_feat

    def forward(self, node_emb):
        # 对节点嵌入进行线性变换并应用GELU激活函数
        nodevec1 = F.gelu(self.alpha * self.lin1(node_emb))
        nodevec2 = F.gelu(self.alpha * self.lin2(node_emb))
        # 使用相似度计算邻接矩阵
        adj = F.relu(
            torch.mm(nodevec1, nodevec2.transpose(1, 0))
            - torch.mm(nodevec2, nodevec1.transpose(1, 0))
        )
        # 如果k小于节点嵌入的维度， 则对邻接矩阵进行top-k操作，保留k个最大的相似度值
        if self.k < node_emb.shape[0]:
            n_nodes = node_emb.shape[0]
            # 初始化掩码矩阵
            mask = torch.zeros(n_nodes, n_nodes).to(node_emb.device)
            mask.fill_(float("0"))
            # 计算邻接矩阵的top-k值
            s1, t1 = (adj + torch.rand_like(adj) * 0.01).topk(self.k, 1)
            # 将top-k值的位置设为1
            mask.scatter_(1, t1, s1.fill_(1))
            # 应用掩码矩阵
            adj = adj * mask
        return adj


class GraphEncoder(nn.Module):
    def __init__(
        self, attn_layers, gnn_layers, gl_layer, node_embs, cls_len, norm_layer=None
    ):
        # 初始化GraphEncoder类，继承自nn.Module
        super(GraphEncoder, self).__init__()
        # 将注意力层列表转换为nn.ModuleList，方便后续处理
        self.attn_layers = nn.ModuleList(attn_layers)
        # 将图神经网络层列表转换为nn.ModuleList，方便后续处理
        self.graph_layers = nn.ModuleList(gnn_layers)
        # 设置图学习层
        self.graph_learning = gl_layer
        # 设置归一化层
        self.norm = norm_layer
        # 设置分类长度
        self.cls_len = cls_len
        # 设置节点嵌入
        self.node_embs = node_embs

    def forward(self, x, attn_mask=None, tau=None, delta=None):
        # x [B, L, D] x的形状为[B, L, D]，其中B是批量大小，L是序列长度，D是特征维度
        attns = []
        # 初始化一个空列表，用于存储每一层的注意力输出
        gcls_len = self.cls_len
        # 获取分类器的长度，即cls_len
        adj = self.graph_learning(self.node_embs)

        # 通过图学习模块计算图的邻接矩阵，输入是节点嵌入
        for i, attn_layer in enumerate(self.attn_layers):
            # 遍历所有的注意力层
            x, attn = attn_layer(x, attn_mask=attn_mask, tau=tau, delta=delta)
            # 前向传播通过当前注意力层，得到输出x和注意力矩阵attn
            attns.append(attn)

            # 将当前层的注意力矩阵添加到列表attns中
            if i < len(self.graph_layers):
                # 如果当前层在图层的范围内
                g = x[:, :gcls_len]
                # 提取x中前gcls_len列的数据，即分类器的部分
                g = rearrange(g, "(b n) p d -> (b p) n d", n=self.node_embs.shape[0])
                # 重塑g的形状，使其适应图层的输入要求
                g = self.graph_layers[i](g, adj) + g
                # 通过当前图层进行前向传播，并加上输入g，实现残差连接
                g = rearrange(g, "(b p) n d -> (b n) p d", p=gcls_len)
                # 将g重塑回原来的形状
                x[:, :gcls_len] = g

                # 将处理后的g替换回x中对应的位置
            if self.norm is not None:
                x = self.norm(x)

        return x, attns


class Model(nn.Module):
    def __init__(self, configs, patch_len=16, stride=8, gc_alpha=1):
        """

        初始化Model类，继承自nn.Module。

        参数说明：
        configs: 配置对象，包含任务相关的配置信息。
        patch_len: int, patch_embedding的patch长度，默认值为16。
        stride: int, patch_embedding的步长，默认值为8。
        gc_alpha: float, 用于梯度裁剪的系数，默认值为1。
        """
        super().__init__()  # 调用父类nn.Module的构造函数，初始化nn.Module
        self.task_name = configs.task_name  # 从配置对象中获取任务名称
        self.seq_len = configs.seq_len  # 从配置对象中获取序列长度
        self.pred_len = configs.pred_len  # 从配置对象中获取预测长度
        padding = stride  # 设置padding为stride的值
        cls_len = configs.cls_len  # 从配置对象中获取cls_len
        gdep = configs.graph_depth  # 从配置对象中获取图深度
        knn = configs.knn  # 从配置对象中获取knn
        embed_dim = configs.embed_dim  # 从配置对象中获取嵌入维度

        # 创建PatchEmbedding对象，用于将输入数据分割成patches并进行嵌入
        self.patch_embedding = PatchEmbedding(
            configs.d_model, patch_len, stride, padding, configs.dropout
        )

        # 创建全局嵌入参数，用于表示全局信息
        self.global_embedding = nn.Parameter(torch.randn(1, cls_len, configs.d_model))

        # 创建GraphEncoder对象，包含多个EncoderLayer和mixprop层，用于处理图结构数据
        self.encoder = GraphEncoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(
                            False,
                            configs.factor,
                            attention_dropout=configs.dropout,
                            output_attention=configs.output_attention,
                        ),
                        configs.d_model,
                        configs.n_heads,
                    ),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation,
                )
                for l in range(configs.e_layers)
            ],
            [
                mixprop(configs.d_model, configs.d_model, gdep)
                for _ in range(configs.e_layers - 1)
            ],
            graph_constructor(configs.enc_in, knn, embed_dim, alpha=gc_alpha),
            nn.Parameter(torch.randn(configs.enc_in, embed_dim), requires_grad=True),
            cls_len,
            norm_layer=nn.LayerNorm(configs.d_model),
        )

        # 预测头
        self.head_nf = configs.d_model * int((configs.seq_len - patch_len) / stride + 2)
        self.head = Flatten_Head(
            configs.enc_in,
            self.head_nf,
            configs.pred_len,
            head_dropout=configs.dropout,
        )

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):

        # 将x_enc的维度从[batch_size, seq_len, features]转换为[batch_size, features, seq_len]
        x_enc = x_enc.permute(0, 2, 1)
        # 对x_enc进行patch嵌入，得到编码输出和变量数
        enc_out, n_vars = self.patch_embedding(x_enc)
        # 获取patch嵌入的长度
        patch_len = enc_out.shape[1]
        # 重复全局嵌入，使其与batch_size相匹配
        global_embeddings = repeat(
            self.global_embedding, "1 n d -> b n d", b=enc_out.shape[0]
        )
        # 将全局嵌入与编码输出拼接在一起
        enc_out = torch.cat([global_embeddings, enc_out], dim=1)

        # 通过编码器处理编码输出，得到编码输出和注意力权重
        enc_out, attns = self.encoder(enc_out)
        # 保留编码输出的最后patch_len部分
        enc_out = enc_out[:, -patch_len:, :]
        # 将编码输出重塑为[batch_size, n_vars, seq_len, features]
        enc_out = torch.reshape(
            enc_out, (-1, n_vars, enc_out.shape[-2], enc_out.shape[-1])
        )
        # 将编码输出的维度从[batch_size, n_vars, features, seq_len]转换为[batch_size, n_vars, seq_len, features]
        enc_out = enc_out.permute(0, 1, 3, 2)

        # 通过预测器处理编码输出，得到解码输出
        dec_out = self.head(enc_out)  # z: [bs x nvars x target_window]
        # 将解码输出的维度从[batch_size, target_window, n_vars]转换为[batch_size, n_vars, target_window]
        dec_out = dec_out.permute(0, 2, 1)

        # 返回解码输出
        return dec_out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        # 检查任务名称是否包含"forecast"（不区分大小写）
        if "forecast" in self.task_name.lower():
            # 调用forecast方法进行预测，传入编码器输入、编码器时间标记、解码器输入和解码器时间标记
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            # 返回预测结果的最后self.pred_len个时间步的输出
            return dec_out[:, -self.pred_len :, :]  # [批量大小, 预测长度, 特征维度]
