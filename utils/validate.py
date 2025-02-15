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
    for iter in tqdm(range(num_query), ascii=True):
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

def CalcMap(rB, qB, retrievalL, queryL):
    # qB: {-1,+1}^{mxq}
    # rB: {-1,+1}^{nxq}
    # queryL: {0,1}^{mxl}
    # retrievalL: {0,1}^{nxl}
    num_query = qB.shape[0]
    map = 0
    # print('------------Calculating MAP------------')
    for iter in tqdm(range(num_query)):
        if num_query==1 :
            gnd = (np.dot(queryL, retrievalL.transpose()) > 0).astype(np.float32)
        else:
            gnd = (np.dot(queryL[iter, :], retrievalL.transpose()) > 0).astype(np.float32)
        tsum = np.sum(gnd)
        if tsum == 0:
            continue
        if num_query == 1:
            hamm = CalcHammingDist(qB, rB)
        else:
            hamm = CalcHammingDist(qB[iter, :], rB)
        ind = np.argsort(hamm)
        gnd = gnd[ind]
        count = np.linspace(1, tsum, int(tsum))

        tindex = np.asarray(np.where(gnd == 1)) + 1.0
        if num_query == 1:
            map_ = np.mean(count[:10] / (tindex[1]))
        else:
            map_ = np.mean(count[:10] / (tindex[0][:10]))

        map = map + map_
    map = map / num_query
    return map
def validate_hash(test_loader, database_loader, model, top_k=None):
  tst_binary, tst_label = compute_result(test_loader, model)
  trn_binary, trn_label = compute_result(database_loader, model)
  mAP = CalcTopMap(trn_binary.numpy(), tst_binary.numpy(), trn_label.numpy(), tst_label.numpy(), top_k)
  return mAP, (tst_binary, trn_binary, tst_label, trn_label)