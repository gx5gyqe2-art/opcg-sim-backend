import json,glob,collections,math,sys
sys.path.insert(0,'.')
from opcg_sim.loop import decks as D
ds=D.user_decks()+D.official_decks()
S=sys.argv[1]
G=[]
for f in sorted(glob.glob(S+"/meta/n5*_p?.json")):
    for g in json.load(open(f))['games']:
        g['wave']=f.split('/')[-1][:3]; G.append(g)
import random
def idx(seed):
    rng=random.Random(seed*7919+17); return rng.randrange(len(ds)), rng.randrange(len(ds))
deck=collections.defaultdict(lambda:[0,0,0,0]) # wins as p1, games p1, wins p2, games p2
lead=collections.defaultdict(lambda:[0,0,0,0])
bad=0
for g in G:
    a,b=idx(g['seed'])
    pass
    w1 = g['winner']=='p1'
    for k,(x,isp1) in enumerate(((a,True),(b,False))):
        win = w1 if isp1 else not w1
        o=0 if isp1 else 2
        deck[x][o]+=win; deck[x][o+1]+=1
        L=ds[x][0]; lead[L][o]+=win; lead[L][o+1]+=1
print("games",len(G),"mismatch",bad, "p1 rate", sum(g['winner']=='p1' for g in G)/len(G))
def row(k,v):
    n=v[1]+v[3]; w=v[0]+v[2]; p=w/n
    se=math.sqrt(p*(1-p)/n)
    return (p, k, n, round(p,3), round(p-1.96*se,3), round(p+1.96*se,3), round(v[0]/max(v[1],1),3), round(v[2]/max(v[3],1),3))
rows=sorted(row(k,v) for k,v in deck.items())
print("decks",len(rows))
print("worst"); [print(r[1:],ds[r[1]][0]) for r in rows[:12]]
print("best"); [print(r[1:],ds[r[1]][0]) for r in rows[-12:]]
lr=sorted(row(k,v) for k,v in lead.items())
print("leaders",len(lr)); [print(r[1:]) for r in lr[:8]]; [print(r[1:]) for r in lr[-8:]]
# first-player diffs
fp=sorted(((v[0]/v[1]-v[2]/v[3]),k,v) for k,v in deck.items() if v[1]>30 and v[3]>30)
print("p1-p2 diff extremes"); [print(round(x[0],3),x[1],ds[x[1]][0],x[2]) for x in fp[:6]+fp[-6:]]
json.dump({str(k):[ds[k][0]]+v for k,v in deck.items()},open(S+"/cut1_deck.json","w"))
