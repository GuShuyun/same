import argparse
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', action='store_true', default="../data/item/",
                        help='dataset')
    parser.add_argument('--baiss', action='store_true', default=True,
                        help='GCN，baiss')
    parser.add_argument('--K', action='store_true', default=1,
                        help='KNN')
    parser.add_argument('--L', action='store_true', default=2,
                        help='lc loss layer')
    parser.add_argument('--gamma', action='store_true', default=1,
                        help='lc loss layer gamma')
    parser.add_argument('--walk_n', action='store_true', default=1,
                        help='random walk number')

    parser.add_argument('--lc_sample', action='store_true', default=50,
                        help='lc loss node sample')

    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='Disables CUDA training.')
    parser.add_argument('--fastmode', action='store_true', default=False,
                        help='Validate during training pass.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed.')
    parser.add_argument('--epochs', type=int, default=1000,
                        help='Number of epochs to train.')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Initial learning rate.')
    parser.add_argument('--weight_decay', type=float, default=5e-4,
                        help='Weight decay (L2 loss on parameters).')

    parser.add_argument('--hidden', type=int, default=256,
                        help='Number of hidden units.')
    parser.add_argument('--out', type=int, default=128,
                        help='Number of hidden units.')

    parser.add_argument('--dropout', type=float, default=0.5,
                        help='Dropout rate (1 - keep probability).')

    parser.add_argument('--t', type=float, default=0.85,
                        help='generate label')
    parser.add_argument('--test_t', type=float, default=0.9,
                        help='test_7')
    parser.add_argument('--pseudo_n', type=int, default=0.5,
                        help='pseudo_n')
    #pseudo

    parser.add_argument('--tau_0', type=float, default=0.5,
                        help='lc tau')
    parser.add_argument('--tau_1', type=float, default=0.5,
                        help='pr tau')

    parser.add_argument('--lambda_1', type=float, default=0.05,
                        help='lambda_1')
    parser.add_argument('--lambda_2', type=float, default=0.5,
                        help='lambda_2')
    parser.add_argument('--lambda_super', type=float, default=0.5,
                        help='lambda_super')
    parser.add_argument('--alfa', type=float, default=0,
                        help='alfa')

    parser.add_argument('--l2', type=float, default=1e-4, help='l2 regularization weight')

    return parser.parse_args()
