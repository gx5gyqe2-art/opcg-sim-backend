import sys,json
sys.path.insert(0,'tests/scripts'); sys.path.insert(0,'.')
from opcg_sim.loop import engine as E, decks as D, driver as DR
E.engine(); db=D.load_db()
kw={"budget":E.PLAY_BUDGET}
if len(sys.argv)>2: kw["commit_hidden_legacy"]=True
spec=E.SeatSpec(None,sims=E.PLAY_SIMS,**kw)
seed=int(sys.argv[1])
la,lb=D.leader_pair(db,seed,"random"); p1,p2=D.build_pair(db,la,lb,seed,"synth_roles")
class Stop(Exception): pass
def obs(game,name,turn,st,out,move):
    if st>=135:
        pend=json.loads(game.pending_json())
        print(st,name,turn,out.get("kind"),pend.get("action"),(pend.get("message") or "")[:60],"commit",[s.get("sig",[None])[0] for s in out.get("commit") or []], json.dumps(move,ensure_ascii=False)[:160],flush=True)
    if st>=157: raise Stop()
try: DR.run_game(seed,{"p1":spec,"p2":spec},p1,p2,observer=obs)
except Stop: pass
