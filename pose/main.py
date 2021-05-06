import matplotlib

from pose.trainer import Trainer

matplotlib.use('Agg')
import argparse
from utils import *

torch.manual_seed(7)
np.random.seed(7)


def parse_opts():
    parser = argparse.ArgumentParser(description='')

    # ========================= Optimizer Parameters ==========================
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--step_size', default=50, type=int)
    parser.add_argument('--weight_decay', default=0, type=float)
    parser.add_argument('--optimizer', type=str, default="Adam")
    parser.add_argument('--momentum', type=float, default=0.9)

    # ========================= Usual Hyper Parameters ==========================
    parser.add_argument('--dropout', default=0.5, type=float)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--exp_name', default="test")
    parser.add_argument('--out_dir', default="exp")
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--grad_clip', type=float, default=0.1)

    # ========================= Network Parameters ==========================
    parser.add_argument('--confidence_threshold', type=float, default=0.1)
    parser.add_argument('--num_classes', type=int, default=7)
    parser.add_argument('--num_total_iterations', type=int, default=10)
    parser.add_argument('--first_layer_size', default=256, type=int)

    # ========================= Training Parameters ==========================
    parser.add_argument('--body_pooling', default="avg", help="how to aggregate the body features sequence")

    parser.add_argument('--weighted_loss', action="store_true", dest="weighted_loss")  # use weighted loss

    parser.add_argument('--use_labels', default=None, type=str,
                        help="if you want to train only body or face models, select 'body' or 'face'")

    args = parser.parse_args()

    return args


def main():
    args = parse_opts()
    b = Trainer(args)
    b.train()


if __name__ == '__main__':
    main()
