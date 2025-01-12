import torch
import torch.nn.functional as F
from tqdm import tqdm

def calc_hamming_dist(b1, b2):
  '''
    Hamming Disctance
  '''
  num_bits = b2.shape[1]
  hamming_dist = 0.5 * (num_bits - b1 @ b2.t())
  return hamming_dist

def calc_map(qb, db, ql, dl, top_k=None):
  '''
    calculate map
  '''
  cnt_q = qb.shape[0]
  _map = 0.0
  for i in tqdm(range(cnt_q)):
    gt = ql[i] @ dl.T > 0
    hd = calc_hamming_dist(qb[i], db)
    idx = torch.argsort(hd, stable=True)
    gt = gt[idx][:top_k]
    cnt = torch.sum(gt).item()

    if cnt == 0:
      continue

    ap = torch.mean(
      torch.linspace(1, cnt, cnt) / (gt.nonzero(as_tuple=True)[0] + 1)
    )
    _map += ap
  _map /= cnt_q
  return _map

def get_codes_and_labels(loader, model, p=None):
  codes = []
  labels_ = []
  with torch.no_grad():
    for images, labels, _ in tqdm(loader):
      images = images.cuda()
      if p is not None:
        images_p = torch.add(images, p)
        code = model(images_p).data.cpu()
        codes.append(code)
        labels_.append(labels)
  return torch.cat(codes).sign(), torch.cat(labels_)

def validate(test_loader, database_loader, model, top_k=None):
  qb, ql = get_codes_and_labels(test_loader, model, top_k)
  db, dl = get_codes_and_labels(database_loader, model, top_k)

  mAP = calc_map(qb, db, ql, dl, top_k=top_k)

  return mAP, (qb, db, ql, dl)

def attack_validate(test_loader, database_loader, tl, model, p=None, top_k=None):
  qb, _ = get_codes_and_labels(test_loader, model, p)
  db, dl = get_codes_and_labels(database_loader, model, None)

  mAP = calc_map(qb, db, tl, dl, top_k=1)

  return mAP, (qb, db, tl, dl)