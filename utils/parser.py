import argparse
import os

def parse_args():
    parser = argparse.ArgumentParser(description="DERK")

    # ===== dataset ===== #
    parser.add_argument("--dataset", nargs="?", default="ml-1m", help="Choose a dataset:[ml-1m, alibaba-fashion, amazon-book]")
    parser.add_argument("--data_path", nargs="?", default="data/", help="Input data path.")

    # ===== train ===== #
    parser.add_argument('--epoch', type=int, default=300, help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=1024, help='batch size')
    parser.add_argument('--test_batch_size', type=int, default=1024, help='batch size')
    parser.add_argument('--dim', type=int, default=128, help='embedding size')
    parser.add_argument('--l2', type=float, default=1e-5, help='l2 regularization weight')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--sim_regularity', type=float, default=1e-4, help='regularization weight for latent factor')
    parser.add_argument("--inverse_r", type=bool, default=True, help="consider inverse relation or not")
    parser.add_argument("--node_dropout", type=bool, default=True, help="consider node dropout or not")
    parser.add_argument("--node_dropout_rate", type=float, default=0.5, help="ratio of node dropout")
    parser.add_argument("--mess_dropout", type=bool, default=True, help="consider message dropout or not")
    parser.add_argument("--mess_dropout_rate", type=float, default=0.1, help="ratio of node dropout")
    parser.add_argument("--batch_test_flag", type=bool, default=True, help="use gpu or not")
    parser.add_argument("--channel", type=int, default=128, help="hidden channels for model")
    parser.add_argument('--Ks', nargs='?', default='[5, 10, 20, 40, 100]', help='Output sizes of every layer')
    parser.add_argument("--n_factors", type=int, default=4, help="number of latent factor for user laten interests")
    parser.add_argument("--ind", type=str, default='distance', help="Independence modeling: mi, distance, cosine")
    parser.add_argument('--test_flag', nargs='?', default='part',
                        help='Specify the test type from {part, full}, indicating whether the reference is done in mini-batch')
    parser.add_argument('--test', action='store_true', help='test model')

    # ===== GPU ===== #
    parser.add_argument("--cuda", type=bool, default=True, help="use gpu or not")
    parser.add_argument("--local_rank", default=os.getenv('LOCAL_RANK', -1), type=int)
    parser.add_argument("--devices", type=str, default="0", help="gpu id of device for ddp train")
    parser.add_argument("--windows", type=int, default=2022, help="windows for ddp train")

    # ===== relation context ===== #
    parser.add_argument('--context_hops', type=int, default=3, help='number of context hops')

    # ===== save model ===== #
    parser.add_argument("--save", type=bool, default=True, help="save model or not")
    parser.add_argument("--out_dir", type=str, default="./weights/", help="output directory for model")

    return parser.parse_args()
