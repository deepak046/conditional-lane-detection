from argparse import ArgumentParser
import sys
import os.path as osp
import mmcv
# Ensure local package imports resolve when running as `python tools/train.py`.
REPO_ROOT = osp.abspath(osp.join(osp.dirname(__file__), '..'))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
from mmdet.apis import inference_detector, init_detector, show_result_pyplot



def main():
    parser = ArgumentParser()
    parser.add_argument('img', help='Image file')
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--score-thr', type=float, default=0.3, help='bbox score threshold')
    parser.add_argument(
        '--out-file',
        default='demo_result.jpg',
        help='Path to save rendered result image')
    args = parser.parse_args()

    # build the model from a config file and a checkpoint file
    model = init_detector(args.config, args.checkpoint, device=args.device)
    # test a single image
    result = inference_detector(model, args.img)
    # render and save results (headless/devcontainer friendly)
    rendered = show_result_pyplot(
        model, args.img, result, score_thr=args.score_thr, show=False)
    mmcv.imwrite(rendered, args.out_file)
    print(f'Saved result to {args.out_file}')


if __name__ == '__main__':
    main()
