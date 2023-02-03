import os
import cv2
import glob
import h5py
import json
import pathlib
import argparse
import numpy as np
import os.path as osp
from scipy.ndimage.filters import gaussian_filter

from src.misc.utilities import vis_density, vis_dot_map


def parse_args():
    parser = argparse.ArgumentParser(
        description='Script for preprocess dataset FDST')

    parser.add_argument('--data-root', type=str,
                        default='~/workspace/datasets/FDST',
                        help='path to the raw dataset')
    parser.add_argument('--destination', type=str,
                        default=None,
                        help='path to the processed data')
    parser.add_argument('--resize-shape', type=int, nargs='+',
                        default=None,
                        help='path to the processed data')

    return parser.parse_args()


def main():
    args = parse_args()
    src_data_root = args.data_root
    dst_data_root = args.destination
    resize_shape = args.resize_shape

    if dst_data_root is None:
        project_path = pathlib.Path(__file__).parent.parent.parent
        dst_data_root = osp.join(project_path, 'processed_data/FDST')
    else:
        dst_data_root = osp.join(osp.abspath(dst_data_root), 'FDST')
    print('Processed files will be saved in: ', osp.abspath(dst_data_root))

    for mode in ['train', 'test']:
        print('-' * 10, f'Processing {mode} data', '-' * 60)

        src_data_dir = osp.join(src_data_root, mode + '_data')
        dst_data_dir = osp.join(dst_data_root, mode + '_data')

        dst_img_dir = osp.join(dst_data_dir, 'imgs')
        dst_den_dir = osp.join(dst_data_dir, 'dens')
        os.makedirs(dst_img_dir, exist_ok=True)
        os.makedirs(dst_den_dir, exist_ok=True)

        scene_list = [p for p in os.listdir(src_data_dir) if not p.startswith('.')]
        assert len(scene_list) % 10 == 0
        scene_list = sorted(scene_list, key=lambda s: int(s))

        for i, scene_name in enumerate(scene_list):
            src_img_path_list = glob.glob(osp.join(src_data_dir, scene_name, '*.jpg'))

            if len(src_img_path_list) == 0:
                print('Error: no images found in ', src_data_root)
                return
            else:
                src_img_path_list = sorted(src_img_path_list,
                                           key=lambda s: int(s.split('.')[0][-3:]))

            for j, src_img_path in enumerate(src_img_path_list):
                print(' [scene: {:2d}/{:2d}] [image: {:3d}/{:3d}]'
                      ' Path: {:s}'.format(
                    i + 1, len(scene_list), j + 1,
                    len(src_img_path_list), src_img_path))

                frame_name = osp.basename(src_img_path)[:3]
                src_ann_path = src_img_path.replace('.jpg', '.json')

                save_name = 's{:03d}_f{:s}'.format(int(scene_name), frame_name)

                dst_img_path = osp.join(dst_img_dir, save_name + '.jpg')
                dst_den_path = osp.join(dst_den_dir, save_name + '.h5')

                img = cv2.imread(src_img_path)
                with open(src_ann_path, 'r') as f:
                    gt = json.load(f)
                gt = list(gt.values())[0]['regions']

                src_h, src_w, _ = img.shape
                if resize_shape is None:
                    dst_h, dst_w = src_h, src_w
                else:
                    dst_h, dst_w = resize_shape[0], resize_shape[1]
                rate_h, rate_w = src_h / dst_h, src_w / dst_w

                # resize img
                img = cv2.resize(img, (dst_w, dst_h))

                # generate dot map & density
                gt_count = len(gt)
                dot_map = np.zeros((dst_h, dst_w))

                for ann in gt:
                    src_rect_x_top_left = ann['shape_attributes']['x']
                    src_rect_y_top_left = ann['shape_attributes']['y']
                    src_rect_width = ann['shape_attributes']['width']
                    src_rect_height = ann['shape_attributes']['height']
                    src_x = src_rect_x_top_left + 0.5 * src_rect_width
                    src_y = src_rect_y_top_left + 0.5 * src_rect_height
                    x = min(int(src_x / rate_w), dst_w)
                    y = min(int(src_y / rate_h), dst_h)
                    dot_map[y, x] = 1

                density = gaussian_filter(dot_map, sigma=5)

                cv2.imwrite(dst_img_path, img)

                with h5py.File(dst_den_path, 'w') as hf:
                    hf['density'] = density
                    hf.close()

                # vis_density(img, density,
                #             save_path=dst_den_path.replace('.h5', '_den.jpg'),
                #             show_img=True)
                #
                # vis_dot_map(img, dot_map,
                #             save_path=dst_den_path.replace('.h5', '_dot.jpg'),
                #             show_img=False)

    print('\nDone!\n')


if __name__ == '__main__':
    main()
