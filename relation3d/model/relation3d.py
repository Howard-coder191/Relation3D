import functools
import gorilla
import pointgroup_ops
import spconv.pytorch as spconv
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_max, scatter_mean, scatter_sum, scatter_softmax
import os
from relation3d.utils import cuda_cast, rle_encode
from .backbone import ResidualBlock, UBlock, MLP
from .loss import Criterion
from .query_decoder import QueryDecoder, SelfAttentionLayer, SelfAttentionLayer
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.preprocessing import LabelEncoder

@gorilla.MODELS.register_module()
class Relation3D(nn.Module):

    def __init__(
        self,
        input_channel: int = 6,
        blocks: int = 5,
        block_reps: int = 2,
        media: int = 32,
        normalize_before=True,
        return_blocks=True,
        pool='mean',
        num_class=18,
        decoder=None,
        criterion=None,
        test_cfg=None,
        norm_eval=False,
        fix_module=[],
    ):
        super().__init__()

        # backbone and pooling
        self.input_conv = spconv.SparseSequential(
            spconv.SubMConv3d(
                input_channel,
                media,
                kernel_size=3,
                padding=1,
                bias=False,
                indice_key='subm1',
            ))
        block = ResidualBlock
        norm_fn = functools.partial(nn.BatchNorm1d, eps=1e-4, momentum=0.1)
        block_list = [media * (i + 1) for i in range(blocks)]
        self.unet = UBlock(
            block_list,
            norm_fn,
            block_reps,
            block,
            indice_key_id=1,
            normalize_before=normalize_before,
            return_blocks=return_blocks,
        )
        self.output_layer = spconv.SparseSequential(norm_fn(media), nn.ReLU(inplace=True))
        self.pool = pool
        self.num_class = num_class

        self.mlp =  nn.Sequential(nn.Linear(2*media, media), nn.ReLU(), nn.Linear(media, media))
        self.pooling_linear = MLP(media, 1, norm_fn=norm_fn, num_layers=3)
        self.pooling_linear1 = MLP(media, 1, norm_fn=norm_fn, num_layers=3)
        self.coords_linear = MLP(3, media, norm_fn=norm_fn, num_layers=3)
        # decoder
        self.decoder = QueryDecoder(**decoder, in_channel=media, num_class=num_class)

        # criterion
        self.criterion = Criterion(**criterion, num_class=num_class)
        self.epoch = 0
        self.test_cfg = test_cfg
        self.norm_eval = norm_eval
        self.loss_out = {'layer_0_sim_loss':0,'layer_1_sim_loss':0,'layer_2_sim_loss':0,'layer_3_sim_loss':0,'layer_4_sim_loss':0,'layer_5_sim_loss':0,'layer_6_sim_loss':0}
        for module in fix_module:
            module = getattr(self, module)
            module.eval()
            for param in module.parameters():
                param.requires_grad = False

    def train(self, mode=True):
        super(Relation3D, self).train(mode)
        if mode and self.norm_eval:
            for m in self.modules():
                # trick: eval have effect on BatchNorm1d only
                if isinstance(m, nn.BatchNorm1d):
                    m.eval()

    def forward(self, batch, mode='loss'):
        if mode == 'loss':
            return self.loss(**batch)
        elif mode == 'predict':
            return self.predict(**batch)

    @cuda_cast
    def loss(self, scan_ids, voxel_coords, p2v_map, v2p_map, spatial_shape, feats, insts, superpoints, coords_float, batch_offsets, sp_instance_labels):
        batch_size = len(batch_offsets) - 1
        voxel_feats = pointgroup_ops.voxelization(feats, v2p_map)
        input = spconv.SparseConvTensor(voxel_feats, voxel_coords.int(), spatial_shape, batch_size)
        sp_coords1 = scatter_mean(coords_float, superpoints, dim=0)  # (B*M, media)
        sp_feats, _, _ = self.extract_feat(input, superpoints, p2v_map, sp_coords1, coords_float)

        out, sp_feats_update_list, _ = self.decoder(sp_feats, sp_coords1, batch_offsets, self.epoch)
        loss_sim = torch.tensor(0.0).cuda()
        sp_instance_label_m_list = []
        for i in range(batch_size):
            sp_instance_label = sp_instance_labels[i]
            sp_instance_label_m = torch.zeros((sp_instance_label.shape[0],sp_instance_label.shape[0])).cuda()
            for j in torch.unique(sp_instance_label):
                a = torch.where(sp_instance_label==j)[0]
                grid_x, grid_y = torch.meshgrid(a, a, indexing='ij')
                sp_instance_label_m[grid_x, grid_y] = 1
            sp_instance_label_m_list.append(sp_instance_label_m)
        loss_out = {}
        for layer_id in range(len(sp_feats_update_list)):
            Sim = F.normalize(sp_feats_update_list[layer_id])@F.normalize(sp_feats_update_list[layer_id]).T
            loss_sim_i = torch.tensor(0.0).cuda()
            for i in range(batch_size):
                sim_sample = Sim[batch_offsets[i]:batch_offsets[i+1],batch_offsets[i]:batch_offsets[i+1]]
                loss_sim_i += F.binary_cross_entropy(torch.clamp((1+sim_sample)/2,0,1), sp_instance_label_m_list[i])
            loss_sim_i = loss_sim_i/batch_size
            loss_out[f'layer_{layer_id}_sim_loss'] = loss_sim_i.item()
            loss_sim += loss_sim_i
        
        loss, loss_dict = self.criterion(out, insts, loss_sim)
        loss_dict.update(loss_out)
        return loss, loss_dict

    @cuda_cast
    def predict(self, scan_ids, voxel_coords, p2v_map, v2p_map, spatial_shape, feats, insts, superpoints, coords_float,
                batch_offsets, sp_instance_labels):
        batch_size = len(batch_offsets) - 1
        voxel_feats = pointgroup_ops.voxelization(feats, v2p_map)
        input = spconv.SparseConvTensor(voxel_feats, voxel_coords.int(), spatial_shape, batch_size)
        sp_coords1 = scatter_mean(coords_float, superpoints, dim=0)  # (B*M, media)
        sp_feats, weightmean, weightmax = self.extract_feat(input, superpoints, p2v_map, sp_coords1, coords_float)
        out, sp_feats_update_list, self_attn = self.decoder(sp_feats, sp_coords1, batch_offsets, self.epoch)
        ret = self.predict_by_feat(scan_ids, out, superpoints, insts)
        return ret

    def predict_by_feat(self, scan_ids, out, superpoints, insts):
        pred_labels = out['labels']
        pred_masks = out['masks']
        pred_scores = out['scores']

        scores = F.softmax(pred_labels[0], dim=-1)[:, :-1]
        nms_score = scores.max(-1)[0].squeeze()
        proposals_pred_f = (pred_masks[0]>0).float()
        intersection = torch.mm(proposals_pred_f, proposals_pred_f.t())  # (nProposal, nProposal), float, cuda
        proposals_pointnum = proposals_pred_f.sum(1)  # (nProposal), float, cuda
        nms_score[proposals_pointnum==0] = 0
        proposals_pn_h = proposals_pointnum.unsqueeze(-1).repeat(1, proposals_pointnum.shape[0])
        proposals_pn_v = proposals_pointnum.unsqueeze(0).repeat(proposals_pointnum.shape[0], 1)
        cross_ious = intersection / (proposals_pn_h + proposals_pn_v - intersection+1e-6)
        pick_idxs = non_max_suppression(cross_ious.cpu().numpy(),nms_score.detach().cpu().numpy(), 0.75)
    
        pred_labels = pred_labels[:,pick_idxs]
        pred_masks[0] = pred_masks[0][pick_idxs]
        scores = scores[pick_idxs]
        labels = torch.arange(
            self.num_class, device=scores.device).unsqueeze(0).repeat(pred_labels.shape[1], 1).flatten(0, 1)
        
        self.test_cfg.topk_insts = min(self.test_cfg.topk_insts, scores.flatten(0, 1).shape[0])
        scores, topk_idx = scores.flatten(0, 1).topk(self.test_cfg.topk_insts, sorted=False)
        labels = labels[topk_idx]
        labels += 1

        topk_idx = torch.div(topk_idx, self.num_class, rounding_mode='floor')
        mask_pred = pred_masks[0]
        mask_pred = mask_pred[topk_idx]
        mask_pred_sigmoid = mask_pred.sigmoid()
        mask_pred = ((mask_pred > 0)).float()   # [n_p, M]
        mask_scores = (mask_pred_sigmoid * mask_pred).sum(1) / (mask_pred.sum(1) + 1e-6)
        scores = scores * mask_scores
        # get mask
        mask_pred = mask_pred[:, superpoints].int()

        # score_thr
        score_mask = scores > self.test_cfg.score_thr
        scores = scores[score_mask]  # (n_p,)
        labels = labels[score_mask]  # (n_p,)
        mask_pred = mask_pred[score_mask]  # (n_p, N)

        # npoint thr
        mask_pointnum = mask_pred.sum(1)
        npoint_mask = mask_pointnum > self.test_cfg.npoint_thr
        scores = scores[npoint_mask]  # (n_p,)
        labels = labels[npoint_mask]  # (n_p,)
        mask_pred = mask_pred[npoint_mask]  # (n_p, N)

        cls_pred = labels.cpu().numpy()
        score_pred = scores.cpu().numpy()
        mask_pred = mask_pred.cpu().numpy()

        pred_instances = []
        for i in range(cls_pred.shape[0]):
            pred = {}
            pred['scan_id'] = scan_ids[0]
            pred['label_id'] = cls_pred[i]
            pred['conf'] = round(score_pred[i], 1)
            # rle encode mask to save memory
            pred['pred_mask'] = rle_encode(mask_pred[i])
            pred_instances.append(pred)

        gt_instances = insts[0].gt_instances
        return dict(scan_id=scan_ids[0], pred_instances=pred_instances, gt_instances=gt_instances)

    def extract_feat(self, x, superpoints, v2p_map, sp_coords, coords_float):
        # backbone
        x = self.input_conv(x)
        x, _ = self.unet(x)
        x = self.output_layer(x)
        x = x.features[v2p_map.long()]  # (B*N, media)

        x_origin = x.clone()
        x = scatter_mean(x_origin, superpoints, dim=0)  # (B*M, media)
        rel_fea_mean = self.pooling_linear((x[superpoints]-x_origin))
        x_mean = scatter_sum(scatter_softmax(rel_fea_mean, superpoints, dim=0) * (x_origin), superpoints, dim=0) 
        x, _ = scatter_max(x_origin, superpoints, dim=0)  # (B*M, media)
        rel_fea_max = self.pooling_linear1((x[superpoints]-x_origin))
        x_max = scatter_sum(scatter_softmax(rel_fea_max, superpoints, dim=0) * (x_origin), superpoints, dim=0)
        x = self.mlp(torch.cat([x_mean, x_max], dim=-1))

        return x, scatter_softmax(rel_fea_mean, superpoints, dim=0), scatter_softmax(rel_fea_max, superpoints, dim=0)



def non_max_suppression(ious, scores, threshold):
    ixs = scores.argsort()[::-1]
    pick = []
    while len(ixs) > 0:
        i = ixs[0]
        pick.append(i)
        iou = ious[i, ixs[1:]]
        remove_ixs = np.where((iou > threshold))[0] + 1
        ixs = np.delete(ixs, remove_ixs)
        ixs = np.delete(ixs, 0)
    return np.array(pick, dtype=np.int32)


def re_order(labels):
    unique_labels = np.unique(labels)
    for i in range(len(unique_labels)):
      labels[labels==unique_labels[i]]=i+1
    return labels