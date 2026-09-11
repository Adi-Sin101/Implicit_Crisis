"""Write validated AI-assisted pre-labels for blind batch 011; never gold labels."""
from pathlib import Path
import pandas as pd

ROOT=Path(__file__).resolve().parents[2]; BLIND=ROOT/'data/gold/annotation/remaining_1837_blind.csv'; OUTDIR=ROOT/'data/gold/annotation/ai_assisted'; OUT=OUTDIR/'batch_011_annotations.csv'
D={
'irf-2d16cbbc81c9':('hard_negative','high','Checks on another person previously in a dark place.'),
'sdcnl-133593af82e1':('explicit_crisis','high','States an intended suicide deadline and desire to act.'),
'sdcnl-4755e95876ad':('non_crisis','high','Requests songs to help cry without crisis meaning.'),
'komati-c969141c0258':('hard_negative','medium','Describes a past attempt but current regret and relief at being alive.'),
'komati-f84ae4f871d9':('non_crisis','high','Inspirational quotation contains no crisis expression.'),
'sdcnl-d592e3a662d9':('hard_negative','medium','Expresses concern about another person’s wellbeing.'),
'komati-edb83599aea3':('non_crisis','high','Invites conversation about unusual topics without crisis content.'),
'irf-1bb9f0843cd4':('explicit_crisis','high','Directly considers suicide and wants to end life.'),
'cams-e27cd838090f':('non_crisis','high','Relationship and social frustration lack crisis-level finality.'),
'irf-01ad64888429':('non_crisis','high','New-year dread is described without crisis meaning.'),
'sdcnl-2bed95f5ea67':('non_crisis','low','Text is too fragmentary to establish crisis meaning.'),
'sdcnl-43f1c88300c8':('explicit_crisis','high','Author explicitly wants to die because of personal emptiness.'),
'suicide_reddit_dataset-a4b9bbe11f94':('explicit_crisis','high','Reports recurrent personal suicidal ideation, planning, and attempts.'),
'irf-c483b8358aa9':('non_crisis','high','Describes feeling unheard and overlooked without crisis finality.'),
'sdcnl-8578043c3b75':('explicit_crisis','high','States intent tonight and describes current access to pills.'),
'komati-39dc4a3f4e4a':('non_crisis','medium','Seeks alternatives to self-harm and asks to feel better.'),
'komati-9fde25838fb0':('hard_negative','high','Explicitly rejects suicide while venting about shame and mistakes.'),
'komati-99f60e51af0f':('non_crisis','low','Urgent request for help lacks enough context to establish crisis meaning.'),
'sdcnl-a3ed749bd719':('implicit_crisis','medium','Hopelessness and wanting to end it convey self-directed finality without overt terms.'),
'komati-d0bd5cbd388c':('explicit_crisis','high','Current desire to die and kill oneself is directly stated.'),
'suicide_reddit_dataset-d9d6d15a90dc':('explicit_crisis','high','Author has a plan for tonight and says they are done.'),
'komati-0afb9a73d2e2':('implicit_crisis','medium','Says they must end it and sees no point in continuing.'),
'suicide_reddit_dataset-87f61c79916c':('hard_negative','medium','Describes a suicide-watch hospitalization but says current state is fine.'),
'komati-03c186dc3315':('non_crisis','high','Asks for gift ideas for relatives without crisis meaning.'),
'sdcnl-1e337b89c2d2':('explicit_crisis','high','Wants to stop living and describes a future suicide scenario.'),
'komati-c9a5fc84e868':('explicit_crisis','high','Describes method research, obtaining a gun, and imminent intent.'),
'komati-9b3f3330d8ed':('explicit_crisis','high','Author is tired of wanting to kill themselves.'),
'irf-3dc6c81c142d':('non_crisis','medium','Depressive symptoms and hope for improvement lack crisis-level finality.'),
'sdcnl-ff43d8a6a775':('non_crisis','low','Describes dark urges and depression but no clear crisis intent.'),
'cams-89df766f6797':('implicit_crisis','high','Burdensomeness and belief others are better off if they die indicate crisis.'),
'komati-3355fe4faa3a':('hard_negative','high','Discusses suicides at school and feeling overwhelmed, not own crisis.'),
'sdcnl-c04936e70e41':('non_crisis','high','Depression symptoms are described without crisis-level meaning.'),
'sdcnl-6f0fadd94ef6':('non_crisis','high','Discusses sleep patterns and depression without crisis meaning.'),
'irf-8a491a74f16f':('non_crisis','high','Seeks help for depression without self-directed crisis.'),
'komati-dd8b275243be':('non_crisis','high','Describes rejection and sadness without crisis content.'),
'irf-79582b4ffbf9':('hard_negative','medium','Refers to reduced past suicidality while discussing present motivation problems.'),
'sdcnl-ed266902fd4a':('explicit_crisis','high','Expresses desire to drown and never be found.'),
'irf-629e120c926a':('implicit_crisis','low','Severe worthlessness and a bleak future suggest possible crisis but finality is indirect.'),
'sdcnl-74d32447ca1f':('non_crisis','high','Wants to leave a painful medical treatment situation, not life itself.'),
'komati-668e69a4e8be':('implicit_crisis','medium','Describes being at a breaking point and needing urgent support.'),
'irf-65f0405cbb61':('hard_negative','high','Discusses another person’s pain and uncertainty about helping her.'),
'komati-069323df365c':('non_crisis','high','Describes recurring disappointment without crisis-level finality.'),
'komati-9972f7dbe16c':('explicit_crisis','high','Past attempt and current wish to die, disappear, and not wake up are described.'),
'irf-efba3ad7d749':('non_crisis','high','New-year greeting contains no crisis meaning.'),
'cams-7192ac4a7aa7':('explicit_crisis','high','Author says they want to die and may act on current suicidal thoughts.'),
'komati-4d4afaae2bfb':('explicit_crisis','high','States an imminent death plan involving pills.'),
'cams-c52e6c4c6617':('explicit_crisis','medium','Current self-inflicted cuts and inability to cope are directly described.'),
'suicide_reddit_dataset-6b5308c0dcb7':('hard_negative','high','Seeks help for a friend with a suicidal history.'),
'komati-0aa43f257321':('hard_negative','high','Reports witnessing another person’s suicide.'),
'komati-8334f29fac13':('hard_negative','high','News-like description of another person trying to kill her husband.'),}

def main():
 seen=set()
 for p in OUTDIR.glob('batch_*_annotations.csv'):
  if p!=OUT: seen.update(pd.read_csv(p,usecols=['id'])['id'].astype(str))
 b=pd.read_csv(BLIND); x=b[~b.id.astype(str).isin(seen)].head(50).copy()
 if len(x)!=50 or set(x.id)!=set(D): raise ValueError('Decisions do not match next 50 blind records.')
 x[['label','confidence','notes']]=[D[i] for i in x.id]
 o=x[['id','text','label','confidence','notes']].reset_index(drop=True)
 if o.id.duplicated().any() or set(o.id)&seen: raise AssertionError('Duplicate or prior ID.')
 if not set(o.label)<= {'explicit_crisis','implicit_crisis','hard_negative','non_crisis'} or not set(o.confidence)<= {'high','medium','low'}: raise AssertionError('Invalid annotation value.')
 OUTDIR.mkdir(parents=True,exist_ok=True); o.to_csv(OUT,index=False,encoding='utf-8'); w=pd.read_csv(OUT)
 if not w.fillna('').astype(str).equals(o.fillna('').astype(str)): raise AssertionError('Saved CSV differs.')
 print(f'AI-assisted pre-label batch 011 written: {len(w)} rows'); print('Labels:\n'+w.label.value_counts().to_string()); print('Confidence:\n'+w.confidence.value_counts().to_string())
if __name__=='__main__': main()
