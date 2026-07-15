import torch, torch.nn as nn, torch.nn.functional as F, numpy as np, json
NCUE=4; DIN=24
torch.manual_seed(123); CUEPROJ=torch.randn(NCUE,DIN)
def gen(B):
    cues=(torch.rand(B,NCUE)<0.4).float(); y=(cues.sum(1)>=2).long(); return cues,y
def to_input(c): return c@CUEPROJ + torch.randn(c.shape[0],DIN)*0.1
class Model(nn.Module):
    def __init__(s,h,depth):
        super().__init__(); L=[nn.Linear(DIN,h),nn.ReLU()]
        for _ in range(depth-1): L+=[nn.Linear(h,h),nn.ReLU()]
        s.body=nn.Sequential(*L); s.head=nn.Linear(h,2)
    def features(s,x):
        h=x
        for m in s.body: h=m(h)
        return h
    def forward(s,x): return s.head(s.features(x))
def train(h,depth,seed,steps=3000):
    torch.manual_seed(seed); m=Model(h,depth); opt=torch.optim.Adam(m.parameters(),lr=3e-3)
    for _ in range(steps):
        c,y=gen(1024); F.cross_entropy(m(to_input(c)),y).backward(); opt.step(); opt.zero_grad()
    return m
def sigs(m):
    out=[]
    for i in range(NCUE):
        c=torch.zeros(500,NCUE); c[:,i]=1
        with torch.no_grad(): out.append(m.features(to_input(c)).mean(0).numpy())
    return np.array(out)
def cka(X,Y):
    X=X-X.mean(0);Y=Y-Y.mean(0)
    return (np.linalg.norm(X.T@Y)**2)/(np.linalg.norm(X.T@X)*np.linalg.norm(Y.T@Y)+1e-12)

archs={"A(w32d1)":(32,1),"B(w48d2)":(48,2),"C(w64d1)":(64,1)}
models={k:[train(h,d,seed=s) for s in range(3)] for k,(h,d) in archs.items()}
S={k:[sigs(m) for m in ms] for k,ms in models.items()}

# cross-architecture CKA (mean over seed pairs)
keys=list(archs)
print("=== Cross-architecture representational similarity (CKA) ===")
res={}
for i in range(len(keys)):
    for j in range(i,len(keys)):
        vals=[cka(a,b) for a in S[keys[i]] for b in S[keys[j]] if not (i==j)]
        if i==j:
            vals=[cka(S[keys[i]][a],S[keys[i]][b]) for a in range(3) for b in range(a+1,3)]
        m=float(np.mean(vals)); res[f"{keys[i]} vs {keys[j]}"]=round(float(m),3)
        print(f"  {keys[i]:9s} vs {keys[j]:9s}: CKA={m:.3f}")

# permutation null: CKA between A and B with B's cue rows shuffled
rng=np.random.default_rng(0)
null=[]
for _ in range(200):
    p=rng.permutation(NCUE)
    null.append(cka(S["A(w32d1)"][0], S["B(w48d2)"][0][p]))
real=cka(S["A(w32d1)"][0],S["B(w48d2)"][0])
pval=float(np.mean(np.array(null)>=real))
print(f"\nA-vs-B real CKA={real:.3f} | permutation-null mean={np.mean(null):.3f} | p={pval:.3f}")
res["permutation_p"]=round(float(pval),3); res["real_AB_cka"]=round(float(real),3); res["null_mean"]=round(float(np.mean(null)),3)
json.dump(res,open("results.json","w"),indent=2); print("saved")
