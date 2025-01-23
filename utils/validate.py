import torch
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np

def compute_result(dataloader, net):
    bs, clses = [], []
    net.eval()
    for img, cls, _ in tqdm(dataloader):
        clses.append(cls)
        bs.append((net(img.cuda())).data.cpu())
    return torch.cat(bs).sign(), torch.cat(clses)

def CalcHammingDist(B1, B2):
    q = B2.shape[1]
    distH = 0.5 * (q - np.dot(B1, B2.transpose()))
    return distH
  
def CalcTopMap(rB, qB, retrievalL, queryL, topk):
    num_query = queryL.shape[0]
    topkmap = 0
    for iter in tqdm(range(num_query)):
        gnd = (np.dot(queryL[iter, :], retrievalL.transpose()) > 0).astype(np.float32)
        hamm = CalcHammingDist(qB[iter, :], rB)
        ind = np.argsort(hamm)
        gnd = gnd[ind]

        tgnd = gnd[0:topk]
        tsum = np.sum(tgnd).astype(int)
        if tsum == 0:
            continue
        count = np.linspace(1, tsum, tsum)

        tindex = np.asarray(np.where(tgnd == 1)) + 1.0
        topkmap_ = np.mean(count / (tindex))
        topkmap = topkmap + topkmap_
    topkmap = topkmap / num_query
    return topkmap

def validate_hash(test_loader, database_loader, model, top_k=None):
  tst_binary, tst_label = compute_result(test_loader, model)
  trn_binary, trn_label = compute_result(database_loader, model)
  mAP = CalcTopMap(trn_binary.numpy(), tst_binary.numpy(), trn_label.numpy(), tst_label.numpy(), top_k)
  return mAP, (tst_binary, trn_binary, tst_label, trn_label)