"""NEXT STEP (implemented): more cue types (12) with correlations so the permutation null is
informative, and test whether cross-architecture universality survives with a real null."""
import torch, torch.nn as nn, torch.nn.functional as F, numpy as np, json
NCUE=12; DIN=48
torch.manual_seed(7)
# correlated cues: a low-rank correlation structure among cues
L=torch.randn(NCUE,4); CORR=(L@L.t()); 
CUEPROJ=torch.randn(NCUE,DIN)
def gen(B):
    base=torch.randn(B,4)@L.t()
    cues=(torch.sigmoid(base+torch.randn(B,NCUE)*0.5)>0.55).float()
    y=(cues.sum(1)>=4).long(); return cues,y
def to_input(c): return c@CUEPROJ+torch.randn(c.shape[0],DIN)*0.1
class M(nn.Module):
    def __init__(s,h,d):
        super().__init__(); L=[nn.Linear(DIN,h),nn.ReLU()]
        for _ in range(d-1): L+=[nn.Linear(h,h),nn.ReLU()]
        s.body=nn.Sequential(*L); s.head=nn.Linear(h,2)
    def feats(s,x):
        h=x
        for m in s.body: h=m(h)
        return h
    def forward(s,x): return s.head(s.feats(x))
def train(h,d,seed):
    torch.manual_seed(seed); m=M(h,d); o=torch.optim.Adam(m.parameters(),lr=3e-3)
    for _ in range(3500):
        c,y=gen(1024); F.cross_entropy(m(to_input(c)),y).backward(); o.step(); o.zero_grad()
    c,y=gen(3000); return m,(m(to_input(c)).argmax(1)==y).float().mean().item()
def sigs(m):
    out=[]
    for i in range(NCUE):
        c=torch.zeros(400,NCUE); c[:,i]=1
        with torch.no_grad(): out.append(m.feats(to_input(c)).mean(0).numpy())
    return np.array(out)
def cka(X,Y):
    X=X-X.mean(0);Y=Y-Y.mean(0)
    return (np.linalg.norm(X.T@Y)**2)/(np.linalg.norm(X.T@X)*np.linalg.norm(Y.T@Y)+1e-12)
mA,aA=train(32,1,1); mB,aB=train(64,2,2)
print(f"acc A={aA:.3f} B={aB:.3f}")
sA,sB=sigs(mA),sigs(mB)
real=cka(sA,sB)
rng=np.random.default_rng(0); null=[cka(sA,sB[rng.permutation(NCUE)]) for _ in range(500)]
p=float(np.mean(np.array(null)>=real))
print(f"12-cue cross-arch CKA={real:.3f} | null mean={np.mean(null):.3f} | p={p:.3f}")
json.dump({"n_cues":NCUE,"accA":round(aA,3),"accB":round(aB,3),
           "real_cka":round(float(real),3),"null_mean":round(float(np.mean(null)),3),"p":round(p,3)},
          open("more_cues_results.json","w"),indent=2); print("saved")
