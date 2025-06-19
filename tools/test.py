import argparse
import gorilla
import torch
from tqdm import tqdm
import os
import time
import sys
current_dir = os.path.dirname(os.path.abspath(sys.argv[0]))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
from relation3d.dataset import build_dataloader, build_dataset
from relation3d.evaluation import ScanNetEval
from relation3d.utils import get_root_logger, save_gt_instances, save_pred_instances
from train import get_model

def get_args():
    parser = argparse.ArgumentParser('SoftGroup')
    parser.add_argument('config', type=str, help='path to config file')
    parser.add_argument('checkpoint', type=str, help='path to checkpoint')
    parser.add_argument('--out', type=str, help='directory for output results')
    args = parser.parse_args()
    return args


def main():
    args = get_args()
    cfg = gorilla.Config.fromfile(args.config)
    gorilla.set_random_seed(cfg.test.seed)
    logger = get_root_logger()

    model_name = cfg.model.pop("name", "Relation3D")
    model = get_model(cfg, model_name)
    cfg.model_name = model_name

    if not os.path.exists(os.path.join(cfg.data.train.data_root, "train")):
        if os.path.exists("/mnt/proj78/ykchen/dataset/scannetv2_spformer/train/scene0000_00_inst_nostuff.pth"):
            cfg.data.train.data_root = "/mnt/proj78/ykchen/dataset/scannetv2_spformer"
            cfg.data.val.data_root = "/mnt/proj78/ykchen/dataset/scannetv2_spformer"
            cfg.data.test.data_root = "/mnt/proj78/ykchen/dataset/scannetv2_spformer"
        elif os.path.exists("/mnt/proj74/xinlai/inst_seg/SPFormer/data/scannetv2/train/scene0000_00_inst_nostuff.pth"):
            cfg.data.train.data_root = "/mnt/proj74/xinlai/inst_seg/SPFormer/data/scannetv2"
            cfg.data.val.data_root = "/mnt/proj74/xinlai/inst_seg/SPFormer/data/scannetv2"
            cfg.data.test.data_root = "/mnt/proj74/xinlai/inst_seg/SPFormer/data/scannetv2"
        elif os.path.exists("/dataset/xinlai/dataset/scannet_spformer/train/scene0000_00_inst_nostuff.pth"):
            cfg.data.train.data_root = "/dataset/xinlai/dataset/scannet_spformer"
            cfg.data.val.data_root = "/dataset/xinlai/dataset/scannet_spformer"
            cfg.data.test.data_root = "/dataset/xinlai/dataset/scannet_spformer"

    logger.info(f'Load state dict from {args.checkpoint}')
    gorilla.load_checkpoint(model, args.checkpoint, strict=False)

    dataset = build_dataset(cfg.data.test, logger)
    dataloader = build_dataloader(dataset, training=False, **cfg.dataloader.test)

    results, scan_ids, pred_insts, gt_insts = [], [], [], []
    sem_labels, ins_labels = [], []
    coords = []
    progress_bar = tqdm(total=len(dataloader))
    pure_inf_time = 0
    with torch.no_grad():
        model.eval()
        for b, batch in enumerate(dataloader):

            if cfg.train.get("use_rgb", True) == False:
                batch['feats'] = batch['feats'][:, 3:]

            if cfg.model_name.startswith("SPFormer"):
                batch.pop("coords_float", "")

            if not cfg.model_name.endswith("no_superpoint"):
                batch.pop("batch_points_offsets", "")
            xyz, _, _, semantic_label, instance_label, _ = dataset.load(dataset.filenames[b])

            if cfg.data.train.type == "scannetv2":
                semantic_label[semantic_label != -100] -= 2
                semantic_label[(semantic_label == -1) | (semantic_label == -2)] = -100
            torch.cuda.synchronize()
            start_time = time.perf_counter()
                  
            result = model(batch, mode='predict')
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start_time
            pure_inf_time += elapsed
            results.append(result)
            sem_labels.append(semantic_label)
            ins_labels.append(instance_label)
            coords.append(xyz)
            progress_bar.update()
        progress_bar.close()
    for res in results:

        scan_ids.append(res['scan_id'])
        pred_insts.append(res['pred_instances'])
        gt_insts.append(res['gt_instances'])

    if not cfg.data.test.prefix == 'test':
        logger.info('Evaluate instance segmentation')
        scannet_eval = ScanNetEval(dataset.CLASSES)
        scannet_eval.evaluate(pred_insts, gt_insts)
    # save output
    if args.out:
        logger.info('Save results')
        nyu_id = dataset.NYU_ID
        save_pred_instances(args.out, 'pred_instance', scan_ids, pred_insts, nyu_id)
        if not cfg.data.test.prefix == 'test':
            save_gt_instances(args.out, 'gt_instance', scan_ids, gt_insts, nyu_id)


if __name__ == '__main__':
    main()
