import torch
import torch.nn as nn
from layers import MLP
from VectorQuantizer import NormEMAVectorQuantizer
from utils import view_quality_from_pred

class Net(nn.Module):
    def __init__(self, d_list, num_classes, z_dim, hidden_dim, num_tokens, codebook_dim, decay, dropout_rate,S,tau):
        super(Net, self).__init__()
        self.d_list = d_list
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate
        self.z_dim = z_dim
        self.hidden_dim = hidden_dim
        self.num_tokens = num_tokens
        self.codebook_dim = codebook_dim
        self.decay = decay
        self.S = S
        self.tau = tau

        self.encoder_list = []
        for v in range(len(d_list)):
            self.encoder_list.append(MLP(self.d_list[v], self.z_dim, self.hidden_dim, dropout_rate=self.dropout_rate, final_act=False, final_norm=False))
        self.encoders = nn.ModuleList(self.encoder_list)

        self.decoder_list = []
        for v in range(len(d_list)):
            self.decoder_list.append(MLP(self.z_dim, self.d_list[v], self.hidden_dim[::-1], dropout_rate=self.dropout_rate, final_act=False, final_norm=False))
        self.decoders = nn.ModuleList(self.decoder_list)

        self.quantize = NormEMAVectorQuantizer(num_embed=num_tokens, embedding_dim=codebook_dim, kmeans_init=True, decay=decay,)

        self.cls_list = []
        for v in range(len(d_list)):
            self.cls_list.append(nn.Linear(z_dim, num_classes))
        self.cls = nn.ModuleList(self.cls_list)


    @torch.jit.ignore
    def no_weight_decay(self):
        return {'quantize.embedding.weight'}

    def forward(self, x_list, mask=None):
        mask = mask.float()

        z_list = []
        for v in range(len(self.d_list)):
            z_list.append(self.encoders[v](x_list[v]))

        packed_z = []
        view_indices = []
        sample_indices = []

        for v in range(len(self.d_list)):
            z_v = z_list[v]
            valid_idx = (mask[:, v] == 1)
            idx = valid_idx.nonzero(as_tuple=True)[0]
            packed_z.append(z_v[valid_idx])
            view_indices.extend([v] * idx.size(0))
            sample_indices.extend(idx.tolist())

        packed_z = torch.cat(packed_z, dim=0)
        num_chunks = self.z_dim // self.codebook_dim
        z_chunks = packed_z.view(-1, num_chunks, self.codebook_dim)

        z_q_all, loss_quantize_total, quantize_indice = self.quantize(z_chunks)
        z_q_all = z_q_all.view(-1, self.z_dim)
        quantize_indice = quantize_indice.view(-1, self.z_dim//self.codebook_dim)

        zq_list = [torch.zeros_like(z_list[v]) for v in range(len(self.d_list))]
        indice_list = [torch.full((z_list[v].shape[0], quantize_indice.shape[1]),
                                  fill_value=-1,
                                  dtype=quantize_indice.dtype,
                                  device=quantize_indice.device)
                       for v in range(len(self.d_list))]
        for i in range(z_q_all.size(0)):
            v = view_indices[i]
            b = sample_indices[i]
            zq_list[v].index_copy_(
                0,
                torch.tensor([b], device=z_q_all.device),
                z_q_all[i].unsqueeze(0)
            )
            indice_list[v][b] = quantize_indice[i]

        xrec_list = []
        for v in range(len(self.d_list)):
            xr_list = []
            for j in range(len(self.d_list)):
                xrs_loc = self.decoders[j](zq_list[v])
                xr_list.append(xrs_loc)
            xrec_list.append(xr_list)

        pred_list = []
        for v in range(len(self.d_list)):
            pred_list.append(torch.sigmoid(self.cls[v](zq_list[v])))

        pred, _ = view_quality_from_pred(pred_list, self.S, mask, tau=self.tau)

        return xrec_list, pred, loss_quantize_total, z_list , pred_list