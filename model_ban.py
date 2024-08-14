# include: graph transformer
from functools import partial

from dgl.nn.pytorch import GATConv, MaxPooling, TAGConv
from layers import Block_CA, Block, Attention, Attention_CA, Mlp
import torch
from torch import nn
import torch.nn.functional as F
from weight_init import trunc_normal_
from BAN import BANLayer
from g_transformer import gt_net_compound, gt_net_protein
from attention_layer import MultiHeadAttention
from dgllife.model.gnn import GCN

if torch.cuda.is_available():
    device = torch.device('cuda')
class PerceiverIODTI(nn.Module):
    # add:gt_layers=10, gt_heads=8
    def __init__(self, embed_dim=256, sequence_conv_dim=256, seq_inpu_dim=20,sequence_conv_kernel=5,depth=2, num_heads=4, mlp_ratio=4., qkv_bias=True, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0.05, norm_layer=partial(nn.LayerNorm, eps=1e-6), global_pool=None, block_layers=Block,
                 act_layer=nn.GELU, attention_block=Attention, mlp_block=Mlp, cross_att_block=Block_CA, input_stages=2,
                 output_stages=2, latent_space=300, dpr_constant=True, mlp_ratio_ca=6.0,  drop_rate_ca=0.0,**kwargs):
        super().__init__()
        self.embedding_dim = embed_dim
        # CNN
        # self.sequence_conv = nn.Conv1d(in_channels=conv_channels,
        #                                out_channels=sequence_conv_dim,
        #                                kernel_size=sequence_conv_kernel,
        #                                padding=0)
        # Transformer
        # self.sequence_transformer = nn.Transformer(d_model=256,nhead=8, num_encoder_layers=6,num_decoder_layers=0,batch_first=True)
        # self.linear = nn.Linear(20,embed_dim)
        # SA
        self.sequence_attention = MultiHeadAttention(input_dim=20,embed_dim=embed_dim,num_heads=2)
        # TAGCN for protein
        self.protein_graph_conv = nn.ModuleList()
        # self.protein_graph_conv.append(TAGConv(74, 74, 4))
        # self.protein_graph_conv.append(TAGConv(74, 74, 4))
        # self.protein_graph_conv.append(TAGConv(41, 41, 4))  # in,out,kernel(邻居范围
        # self.protein_graph_conv.append(GCN(41, hidden_feats=[embed_dim//2], batchnorm=[False],allow_zero_in_degree=True))
        # self.protein_graph_conv.append(GCN(embed_dim // 2, hidden_feats=[embed_dim], batchnorm=[False],allow_zero_in_degree=True))
        # 一层GCN
        # self.protein_graph_conv.append(GCN(41, hidden_feats=[embed_dim], batchnorm=[False], allow_zero_in_degree=True))

        # TAGCN for ligand
        self.ligand_graph_conv = nn.ModuleList()
        # self.ligand_graph_conv.append(TAGConv(44, 44, 4))
        # self.ligand_graph_conv.append(GCN(44, hidden_feats=[embed_dim//2], batchnorm=[False],allow_zero_in_degree=True))
        # self.ligand_graph_conv.append(GCN(embed_dim // 2, hidden_feats=[embed_dim], batchnorm=[False],allow_zero_in_degree=True))
        # 一层GCN
        # self.ligand_graph_conv.append(GCN(44, hidden_feats=[embed_dim], batchnorm=[False], allow_zero_in_degree=True))

        # GraphTransformer for protein
        self.protein_graph_conv = gt_net_protein.GraphTransformer(device, n_layers=2, node_dim=41, edge_dim=5, hidden_dim=embed_dim,
                             out_dim=embed_dim, n_heads=num_heads, in_feat_dropout=0.0, dropout=drop_rate,
                             pos_enc_dim=8)
        # GraphTransformer for ligand
        self.ligand_graph_conv = gt_net_compound.GraphTransformer(device, n_layers=2, node_dim=44, edge_dim=10, hidden_dim=embed_dim,
                             out_dim=embed_dim, n_heads=num_heads, in_feat_dropout=0.0, dropout=drop_rate,
                             pos_enc_dim=8)



        # latent_space: 特征数
        self.latent_query = nn.Parameter(torch.zeros(1, latent_space, embed_dim))
        dpr = [drop_path_rate for i in range(depth)]
        # self-attention
        self.blocks = nn.ModuleList([
            block_layers(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                act_layer=act_layer, Attention_block=attention_block, Mlp_block=mlp_block)
            for i in range(depth)])
        # ca:cross-attention
        self.blocks_ca_input = nn.ModuleList([
            Block_CA(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio_ca, qkv_bias=qkv_bias, qk_scale=qk_scale,
                     drop=drop_rate_ca, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                     act_layer=act_layer, Attention_block=cross_att_block, Mlp_block=mlp_block) for i in
            range(input_stages)
        ])

        self.blocks_ca_output = nn.ModuleList([
            Block_CA(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                     drop=0.0, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                     act_layer=act_layer, Attention_block=cross_att_block, Mlp_block=mlp_block) for i in
            range(output_stages)
        ])

        self.norm = norm_layer(embed_dim)
        self.feature_info = [dict(num_chs=embed_dim, reduction=0, module='head')]
        self.head = nn.Linear(embed_dim, 1)
        trunc_normal_(self.latent_query, std=.02)

        self.apply(self._init_weights)

        # # 更改为 BANLayer 模块
        # self.ban_module = BANLayer(
        #     v_dim = sequence_conv_dim,  # 请替换为实际的全局特征维度
        #     q_dim = embed_dim,  # 请替换为实际的局部特征维度
        #     h_dim = hidden_dim,
        #     h_out = output_dim
        #     # 其他 BANLayer 模块的参数根据实际情况设置
        # )

        self.ban = BANLayer(v_dim=embed_dim, q_dim=embed_dim, h_dim=embed_dim, h_out=2)
        # 对输出特征分类
        self.mlp_classifier = MLPDecoder(in_dim=embed_dim, hidden_dim=512, out_dim=128, binary=1)

        # 增加超参α
        # self.alpha = nn.Parameter(torch.tensor(0.5), requires_grad=True)
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'latent_query'}

    def get_classifier(self):
        return self.head

    # def forward(self, g):
    #     feature_protein = g[0].ndata['h']
    #     feature_smile = g[1].ndata['h']
    #
    #     # 处理蛋白质图结构
    #     for module in self.protein_graph_conv:
    #         feature_protein = F.relu(module(g[0], feature_protein))
    #         feature_protein = F.dropout(feature_protein, 0.05)
    #         # feature_protein = F.normalize(feature_protein)
    #     # 处理配体图结构
    #     for module in self.ligand_graph_conv:
    #         feature_smile = F.relu(module(g[1], feature_smile))
    #         feature_smile = F.dropout(feature_smile, 0.05)
    #         # feature_smile = F.normalize(feature_smile)
    #     # 处理蛋白质序列结构
    #     seq_representation = F.relu(self.sequence_conv())
    #     seq_representation = F.dropout(seq_representation, 0.05)
    #
    #     pool_ligand = MaxPooling()
    #     pool_protein = MaxPooling()
    #     protein_rep = pool_protein(g[0], feature_protein).view(1, -1, self.embedding_dim)
    #     ligand_rep = pool_ligand(g[1], feature_smile).view(1, -1, self.embedding_dim)
    #
    #     # 将图卷积后的蛋白质特征与序列信息拼接在一起
    #     co_rep = torch.cat([protein_rep, seq_representation.unsqueeze(0)], dim=1)
    #
    #     x = self.latent_query.expand(1, -1, -1)
    #     for i, blk in enumerate(self.blocks_ca_input):
    #         # x, _ = blk(x, protein_rep)
    #         x, _ = blk(x, co_rep)
    #
    #     for i, blk in enumerate(self.blocks):
    #         x, _ = blk(x)
    #
    #     for i, blk in enumerate(self.blocks_ca_output):
    #         x, _ = blk(x, ligand_rep)
    #
    #     x = self.norm(x)
    #     x = torch.mean(x, dim=1)
    #     return torch.sigmoid(self.head(x))
    def forward(self, inp):
        prot_graph, target_graph, pdb_sequences = inp
        # print("prot_graph shape:", prot_graph.ndata['h'].shape)
        # print("pdb_sequences shape:", pdb_sequences.shape)
        feature_protein = prot_graph.ndata['h'] # 节点特征
        # print("蛋白质节点特征：",feature_protein.shape)
        feature_smile = target_graph.ndata["h"]
        # print("化合物节点特征：", feature_smile.shape)

        # TAGCN for protein
        # for module in self.protein_graph_conv:
        #     feature_protein = F.relu(module(prot_graph, feature_protein))
        #     feature_protein = F.dropout(feature_protein, 0.05)
        # GT for protein
        feature_protein = self.protein_graph_conv(prot_graph)
        feature_protein = F.dropout(feature_protein, 0.05)

        # TAGCN for compound
        # for module in self.ligand_graph_conv:
        #     feature_smile = F.relu(module(target_graph, feature_smile))
        #     feature_smile = F.dropout(feature_smile, 0.05)
        # GT for compound
        feature_smile = self.ligand_graph_conv(target_graph)
        # print("feature_smile shape:", feature_smile)
        feature_smile = F.dropout(feature_smile, 0.05)

        # CNN
        # print("pdb_sequences shape before convolution:", pdb_sequences.shape)
        # seq_representation = F.relu(self.sequence_conv(pdb_sequences))
        # TF
        # src = pdb_sequences.transpose(0,1).unsqueeze(0)
        # # print("src shape:", src.shape)
        # src_x = self.linear(src)
        # # print("src_x shape:", src_x.shape)
        # tgt = pdb_sequences.transpose(0,1).unsqueeze(0)
        # # print("tgt shape:", tgt.shape)
        # tgt_x = self.linear(tgt)
        # # print("tgt_x shape:", tgt_x.shape)
        # seq_representation = F.relu(self.sequence_transformer(src_x,tgt_x))
        # # print("seq_representation:",seq_representation)
        # seq_representation = F.dropout(seq_representation, 0.05)

        # self-attention
        pdb_sequences = pdb_sequences.transpose(0,1)
        # print(pdb_sequences.shape)
        pdb_sequences = pdb_sequences.unsqueeze(0)
        # print(pdb_sequences.shape)
        seq_representation = F.relu(self.sequence_attention(pdb_sequences))
        # print("seq_representation shape after tf:", seq_representation.shape)


        pool_ligand = MaxPooling()
        pool_protein = MaxPooling()

        # 经过图卷积和池化后的特征表示
        protein_rep = pool_protein(prot_graph, feature_protein).view(1, -1, self.embedding_dim)
        # print("after pooling protein_rep shape:", protein_rep.shape)
        ligand_rep = pool_ligand(target_graph, feature_smile).view(1, -1, self.embedding_dim)
        # print("after pooling ligand_rep shape:", ligand_rep.shape)
        # adaptive_avg_pool1d：batch_size,channels,len
        # seq_rep = F.adaptive_avg_pool1d(seq_representation.unsqueeze(0), output_size=protein_rep.shape[1]).transpose(1,2)
        # print("after pooling seq_rep shape:", seq_rep.shape)
        # 将图卷积后的蛋白质特征与序列信息拼接在一起
        # co_rep = torch.cat([protein_rep, seq_representation.unsqueeze(0).transpose(1,2)], dim=1)
        # 图与序列取平均
        # co_rep = (protein_rep + seq_rep) / 2.0
        # print("co_rep shape:", co_rep.shape)
        # 调用 BANLayer 模块进行特征融合（不合理
        # co_rep, att_maps = self.ban_module(seq_representation.unsqueeze(0).transpose(1,2), protein_rep)
        # 加权重 超参数
        # co_rep =  self.alpha * protein_rep + (1 - self.alpha) * seq_rep
        # print("co_rep shape:", co_rep.shape)
        # 使用cross-attention for 特征融合:seq(Q) poc(K,V)
        # seq_representation = seq_representation.unsqueeze(0).transpose(1,2)
        for i, blk in enumerate(self.blocks_ca_input):
            co_rep, _ = blk(seq_representation,protein_rep)
        # +latent-space
        # x = self.latent_query.expand(1, -1, -1)
        # for i, blk in enumerate(self.blocks_ca_input):
        #     x, _ = blk(x, co_rep)
        # self-attention for latent_space
        # for i, blk in enumerate(self.blocks):
        #     x, _ = blk(x)
        # for i, blk in enumerate(self.blocks):
        #     x, _ = blk(x)
        #
        # for i, blk in enumerate(self.blocks_ca_output):
        #     x, _ = blk(x, ligand_rep)
        #
        # x = self.norm(x)
        # x = torch.mean(x, dim=1)
        # return torch.sigmoid(self.head(x))

        # -latent_space
        # for i, blk in enumerate(self.blocks):
        #     co_rep, _ = blk(co_rep)
        #
        # for i, blk in enumerate(self.blocks_ca_output):
        #     co_rep, _ = blk(co_rep, ligand_rep)
        #
        # co_rep = self.norm(co_rep)
        # co_rep = torch.mean(co_rep, dim=1)
        # return torch.sigmoid(self.head(co_rep))

        # +BAN
        # print("BAN")
        f, att = self.ban(ligand_rep, co_rep)
        # print(f.shape)
        score = self.mlp_classifier(f)
        # print(score)
        return torch.sigmoid(score)

class MLPDecoder(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, binary=1):
        super(MLPDecoder, self).__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        # self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn1 = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        # self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.LayerNorm(hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, out_dim)
        # self.bn3 = nn.BatchNorm1d(out_dim)
        self.bn3 = nn.LayerNorm(out_dim)
        self.fc4 = nn.Linear(out_dim, binary)

    def forward(self, x):
        x = self.bn1(F.relu(self.fc1(x)))
        x = self.bn2(F.relu(self.fc2(x)))
        x = self.bn3(F.relu(self.fc3(x)))
        x = self.fc4(x)
        return x




