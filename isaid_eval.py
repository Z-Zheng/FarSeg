import argparse
import logging
import torch
import numpy as np
import simplecv as sc
from data.isaid import COLOR_MAP
from data.isaid import ImageFolderDataset
from concurrent.futures import ProcessPoolExecutor
from tensorboardX import SummaryWriter
from module import farseg
from torch.utils.data.dataloader import DataLoader
from simplecv.api.preprocess import comm
from simplecv.api.preprocess import segm
from tqdm import tqdm
from simplecv.data.preprocess import sliding_window


class SegmSlidingWinInference(object):
    def __init__(self):
        super(SegmSlidingWinInference, self).__init__()
        self._h = None
        self._w = None
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    def patch(self, input_size, patch_size, stride, transforms=None):
        """ divide large image into small patches.

        Returns:

        """
        self.wins = sliding_window(input_size, patch_size, stride)
        self.transforms = transforms
        return self

    def merge(self, out_list):
        pred_list, win_list = list(zip(*out_list))
        num_classes = pred_list[0].size(1)
        res_img = torch.zeros(pred_list[0].size(0), num_classes, self._h, self._w, dtype=torch.float32)
        res_count = torch.zeros(self._h, self._w, dtype=torch.float32)

        for pred, win in zip(pred_list, win_list):
            res_count[win[1]:win[3], win[0]: win[2]] += 1
            res_img[:, :, win[1]:win[3], win[0]: win[2]] += pred.cpu()

        avg_res_img = res_img / res_count

        return avg_res_img

    def forward(self, model, image_np, **kwargs):
        assert self.wins is not None, 'patch must be performed before forward.'
        # set the image height and width
        self._h, self._w, _ = image_np.shape
        return self._forward(model, image_np, **kwargs)

    def _forward(self, model, image_np, **kwargs):
        self.device = kwargs.get('device', self.device)
        size_divisor = kwargs.get('size_divisor', None)
        assert self.wins is not None, 'patch must be performed before forward.'
        out_list = []
        for win in tqdm(self.wins):
            x1, y1, x2, y2 = win
            image = image_np[y1:y2, x1:x2, :].astype(np.float32)
            if self.transforms is not None:
                image = self.transforms(image)
            h, w = image.shape[2:4]
            if size_divisor is not None:
                image = sc.preprocess.function.th_divisible_pad(image, size_divisor)
            image = image.to(self.device)
            with torch.no_grad():
                out = model(image)
            if size_divisor is not None:
                out = out[:, :, :h, :w]
            out_list.append((out.cpu(), win))
            torch.cuda.empty_cache()
        self.wins = None

        return self.merge(out_list)


parser = argparse.ArgumentParser()
parser.add_argument('--config_path', default=None, type=str,
                    help='path to config file')
parser.add_argument('--ckpt_path', default=None, type=str,
                    help='path to model directory')
parser.add_argument('--image_dir', default=None, type=str,
                    help='path to image dir')
parser.add_argument('--mask_dir', default=None, type=str,
                    help='path to mask dir')
parser.add_argument('--vis_dir', default=None, type=str,
                    help='path to vis_dir')
parser.add_argument('--log_dir', default=None, type=str,
                    help='path to log')
parser.add_argument('--patch_size', default=896, type=int,
                    help='patch size')

logger = logging.getLogger('SW-Infer')
logger.setLevel(logging.INFO)


def run():
    args = parser.parse_args()

    model, global_step = sc.infer_tool.build_and_load_from_file(args.config_path, args.ckpt_path)
    model.to(torch.device('cuda'))
    segm_helper = SegmSlidingWinInference()

    ppe = ProcessPoolExecutor(max_workers=4)

    dataset = ImageFolderDataset(image_dir=args.image_dir, mask_dir=args.mask_dir)

    palette = np.asarray(list(COLOR_MAP.values())).reshape((-1,)).tolist()

    viz_op = sc.viz.VisualizeSegmm(args.vis_dir, palette=palette)
    miou_op = sc.metric.NPmIoU(num_classes=16, logdir=args.log_dir)

    image_trans = comm.Compose([
        segm.ToTensor(True),
        comm.THMeanStdNormalize((123.675, 116.28, 103.53), (58.395, 57.12, 57.375)),
        comm.CustomOp(lambda x: x.unsqueeze(0))
    ])

    for idx, blob in enumerate(
            DataLoader(dataset, 1, shuffle=False, pin_memory=True, num_workers=4, collate_fn=lambda x: x)):
        image, mask, filename = blob[0]

        h, w = image.shape[:2]
        logging.info('Progress - [{} / {}] size = ({}, {})'.format(idx + 1, len(dataset), h, w))
        seg_helper = segm_helper.patch((h, w), patch_size=(args.patch_size, args.patch_size), stride=512,
                                       transforms=image_trans)

        out = seg_helper.forward(model, image, size_divisor=32)

        out = out.argmax(dim=1)

        if mask is not None:
            miou_op.forward(mask, out)
        ppe.submit(viz_op, out.numpy(), filename)

    ppe.shutdown()
    ious, miou = miou_op.summary()

    # tensorboard
    sw = SummaryWriter(logdir=args.log_dir)

    sw.add_scalar('eval-miou/miou', miou, global_step=global_step)
    sw.add_scalar('eval-miou/miou-fg', ious[1:].mean(), global_step=global_step)
    for name, iou in zip(list(COLOR_MAP.keys()), ious):
        sw.add_scalar('eval-ious/{}'.format(name), iou, global_step=global_step)

    sw.close()


if __name__ == '__main__':
    run()
