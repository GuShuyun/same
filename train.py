from __future__ import division
from __future__ import print_function

import time
import argparse
import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F
import torch.optim as optim

from code.utils import load_data
from code.models import GNN
from code.parser import parse_args
from sklearn.metrics.pairwise import cosine_similarity as cos
import warnings
warnings.filterwarnings('ignore')
def test(model, pic_feature, adj0_pre, text_feature, adj1_pre, testset, test_t):
    model.eval()
    x_all = model.generate(pic_feature, adj0_pre, text_feature, adj1_pre)
    t = torch.tensor(np.array(testset).flatten())
    x_t = x_all[t]

    scores = torch.tensor(cos(x_t.detach().numpy()))
    # scores = torch.cosine_similarity(output, output, dim=-1)
    scores = scores - torch.diag_embed(torch.diag(scores))
    scores = torch.triu(scores, diagonal=0)
    ind = torch.nonzero(scores > 0.95).squeeze().tolist()
    count = 0
    for i in ind:
        if ([int(t[i[0]]), int(t[i[1]])] or [int(t[i[1]]), int(t[i[0]])]) in testset:
            count = count + 1
    R = count / len(testset)
    P = count / len(ind)
    print("count: ",count )
    print("len(testset): ", len(testset))
    print("len(ind): ", len(ind))

    return R, P

if __name__ == '__main__':
    # Training settings
    args = parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
    # Load data
    adj0_pre, adj1_pre, pic_feature, text_feature, trainset, testset = load_data(args.dataset, args.K)
    adj0 = adj0_pre
    adj1 = adj1_pre
    #testset = testset.tolist()

    model = GNN(input=pic_feature.shape[1],
                hidden=args.hidden,
                output=args.out,
                dropout=args.dropout,
                baiss=args.baiss,
                args=args)
    optimizer = optim.Adam(model.parameters(),
                           lr=args.lr, weight_decay=args.weight_decay)
    if args.cuda:
        model.cuda()
        # model_t.cuda()
        pic_feature = pic_feature.cuda()
        text_feature = text_feature.cuda()
        adj0 = adj0.cuda()
        adj1 = adj1.cuda()
        trainset = trainset.cuda()
        testset = testset.cuda()

    t_total = time.time()

    for epoch in range(args.epochs):
        t = time.time()
        model.train()
        optimizer.zero_grad()
        x_all, loss_train = model(pic_feature, adj0, text_feature, adj1, trainset)
        print(epoch, loss_train)

        t = torch.tensor(np.array(testset).flatten())
        x_t = x_all[t]
        scores = torch.tensor(cos(x_t.detach().numpy()))
        # scores = torch.cosine_similarity(output, output, dim=-1)
        scores = scores - torch.diag_embed(torch.diag(scores))
        scores = torch.triu(scores, diagonal=0)
        ind = torch.nonzero(scores > 0.85).squeeze().tolist()
        count = 0
        for i in ind:
            if ([int(t[i[0]]), int(t[i[1]])] or [int(t[i[1]]), int(t[i[0]])]) in testset:
                count = count + 1
        R = count / len(testset)
        P = count / len(ind)
        print("count: ", count)
        print("len(testset): ", len(testset))
        print("len(ind): ", len(ind))
        print(R, P)

        loss_train.backward()
        optimizer.step()

        #generate(self, x0, adj0, x1, adj1):

    R, P= test(model, pic_feature, adj0_pre, text_feature, adj1_pre, testset, args.test_t)
    print(R, P)