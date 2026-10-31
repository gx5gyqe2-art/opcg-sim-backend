import json,glob,sys,random,math
def load(d,c):
    out={}
    for f in glob.glob(f"n1_results/{d}/{c}_w*.jsonl"):
        for l in open(f):
            r=json.loads(l); out[r["seed"]]=sum(r["games"])/2
    return out
a,b=sys.argv[1],sys.argv[2]
for c in ("real","mirror","synth2"):
    A,B=load(a,c),load(b,c); ks=sorted(set(A)&set(B))
    d=[A[k]-B[k] for k in ks]; n=len(d); m=sum(d)/n
    sd=math.sqrt(sum((x-m)**2 for x in d)/(n-1)); se=sd/math.sqrt(n)
    def elo(p): return -400*math.log10(1/p-1)
    pa=sum(A[k] for k in ks)/n; pb=sum(B[k] for k in ks)/n
    print(f"{c}: n={n} {a} {pa:.4f} {b} {pb:.4f} diff_wr {m:+.4f} ±{1.96*se:.4f}  Elo diff ≈ {elo(pa)-elo(pb):+.1f} [{elo(pa)-elo(pa-m+1.96*se)+0:+.1f}..]")
