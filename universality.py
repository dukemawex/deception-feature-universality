"""Feature-universality pilot (DeepMind/Sharkey proposal).
Question: do TWO different models, trained on the same 'deception-cue' task, develop SIMILAR
internal building blocks for the cue concepts, or divergent ones?

Toy 'deception detection': inputs encode 4 cue types (urgency, impersonation, false-authority,
trust-manipulation) as latent factors mixed into a feature vector; label = phishing if >=2 cues.
We train two models of DIFFERENT width/depth, decompose each (SPD), extract per-cue component
signatures, and measure cross-model representational similarity (CKA + matched-signature corr).
"""
import torch, torch.nn as nn, torch.nn.functional as F, numpy as np, json

NCUE=4; DIN=24
def gen(B, seed=None):
    if seed is not None: torch.manual_seed(seed)
    cues=(torch.rand(B,NCUE)<0.4).float()           # which cues present
    # each cue projects onto a fixed random subspace of the input
    y=(cues.sum(1)>=2).long()
    return cues, y

# fixed cue->input projection (shared 'world'): both models see the same inputs
torch.manual_seed(123)
CUEPROJ=torch.randn(NCUE,DIN)
def to_input(cues): 
    return cues@CUEPROJ + torch.randn(cues.shape[0],DIN)*0.1

class Model(nn.Module):
    def __init__(s,h=32,depth=1):
        super().__init__()
        layers=[nn.Linear(DIN,h),nn.ReLU()]
        for _ in range(depth-1): layers+=[nn.Linear(h,h),nn.ReLU()]
        s.body=nn.Sequential(*layers); s.head=nn.Linear(h,2); s.h=h; s.rep_layer=0
    def features(s,x):
        h=x
        for m in s.body: h=m(h)
        return h
    def forward(s,x): return s.head(s.features(x))

def train(h,depth,seed,steps=3000):
    torch.manual_seed(seed); m=Model(h,depth); opt=torch.optim.Adam(m.parameters(),lr=3e-3)
    for _ in range(steps):
        cues,y=gen(1024); x=to_input(cues)
        F.cross_entropy(m(x),y).backward(); opt.step(); opt.zero_grad()
    cues,y=gen(4000); acc=(m(to_input(cues)).argmax(1)==y).float().mean().item()
    return m,acc

# two DIFFERENT architectures
mA,accA=train(h=32,depth=1,seed=1)
mB,accB=train(h=48,depth=2,seed=2)
print(f"Model A (w32,d1) acc={accA:.3f} | Model B (w48,d2) acc={accB:.3f}")

# per-cue representation signature: mean hidden activation when ONLY cue i is present
def cue_signatures(m):
    sigs=[]
    for i in range(NCUE):
        cues=torch.zeros(500,NCUE); cues[:,i]=1.0
        with torch.no_grad(): h=m.features(to_input(cues))
        sigs.append(h.mean(0).numpy())
    return np.array(sigs)   # [NCUE, h]
sA=cue_signatures(mA); sB=cue_signatures(mB)

# Linear CKA between the two models' cue-signature spaces (architecture-agnostic similarity)
def cka(X,Y):
    X=X-X.mean(0); Y=Y-Y.mean(0)
    xy=np.linalg.norm(X.T@Y)**2; xx=np.linalg.norm(X.T@X); yy=np.linalg.norm(Y.T@Y)
    return xy/(xx*yy+1e-12)
cka_val=cka(sA,sB)

# matched-cue correlation: does cue i in A correlate with cue i in B (via their similarity matrices)?
def sim_matrix(S):
    Sn=S/(np.linalg.norm(S,axis=1,keepdims=True)+1e-9); return Sn@Sn.T
simA=sim_matrix(sA); simB=sim_matrix(sB)
iu=np.triu_indices(NCUE,1)
struct_corr=np.corrcoef(simA[iu],simB[iu])[0,1]

print(f"\n=== Feature universality across architectures ===")
print(f"Linear CKA (cue representations A vs B): {cka_val:.3f}   (1=identical structure)")
print(f"Cue-similarity-structure correlation:    {struct_corr:.3f}   (do cues relate the same way?)")
# baseline: shuffle B's cues
rng=np.random.default_rng(0); perm=rng.permutation(NCUE)
struct_shuf=np.corrcoef(simA[iu], sim_matrix(sB[perm])[iu])[0,1]
print(f"shuffled-cue baseline:                   {struct_shuf:.3f}")
json.dump({"accA":round(accA,3),"accB":round(accB,3),
           "cka":round(float(cka_val),3),"structure_corr":round(float(struct_corr),3),
           "shuffled_baseline":round(float(struct_shuf),3)}, open("results.json","w"),indent=2)
np.save("sigA.npy",sA); np.save("sigB.npy",sB)
print("saved results.json")
