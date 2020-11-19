from torch.utils.data import Dataset

from simplecv.data.preprocess import sliding_window
from PIL import Image
import simplecv as sc

DEFAULT_PATCH_CONFIG = dict(
    patch_size=896,
    stride=512,
)


class PatchBasedDataset(Dataset):
    def __init__(self,
                 image_dir,
                 mask_dir,
                 patch_config=DEFAULT_PATCH_CONFIG,
                 transforms=None):
        super(PatchBasedDataset, self).__init__()
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.patch_cfg = patch_config
        self.transforms = sc.preprocess.comm.Compose(transforms)
        self._data_list = []
        self.patch()

    def __getitem__(self, idx):
        image_path, mask_path, win = self._data_list[idx]
        img = Image.open(image_path).convert('RGB')
        if mask_path is not None:
            # train mode
            mask = Image.open(mask_path).convert('RGB')
        else:
            # test mode
            mask = None

        img = img.crop(win)
        mask = mask.crop(win)

        if self.transforms is not None:
            img, mask = self.transforms(img, mask)

        return img, dict(cls=mask)

    def __len__(self):
        return len(self._data_list)

    def generate_path_pair(self):
        return NotImplementedError

    def patch(self, patch_config=None):
        if patch_config is not None:
            self.patch_cfg = patch_config

        patch_size = self.patch_cfg.get('patch_size', DEFAULT_PATCH_CONFIG['patch_size'])
        stride = self.patch_cfg.get('stride', DEFAULT_PATCH_CONFIG['stride'])

        self._data_list.clear()
        for path_pair in self.generate_path_pair():
            im_path, mask_path = path_pair
            image = Image.open(im_path)
            wins = sliding_window(input_size=(image.height, image.width), kernel_size=(patch_size, patch_size),
                                  stride=stride)

            self._data_list += [(im_path, mask_path, win) for win in wins]
