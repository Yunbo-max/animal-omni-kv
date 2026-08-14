#!/usr/bin/env python3
"""Evaluate a saved Thinker LoRA adapter with deterministic generation."""
from __future__ import annotations
import argparse,csv
from pathlib import Path
import yaml
from peft import PeftModel
from animal_omni.qwen_runner import QwenThinkerRunner
from animal_omni.metrics import normalize_label

def main():
 p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--manifest',type=Path,required=True);p.add_argument('--model-id',required=True);p.add_argument('--adapter',type=Path,required=True);p.add_argument('--split',default='test');p.add_argument('--output',type=Path,required=True);a=p.parse_args();cfg=yaml.safe_load(a.config.read_text());labels=cfg['dataset']['labels'];rows=[r for r in csv.DictReader(a.manifest.open()) if r['split']==a.split];runner=QwenThinkerRunner(a.model_id);runner.model.thinker=PeftModel.from_pretrained(runner.model.thinker,a.adapter);out=[]
 for i,r in enumerate(rows,1):
  raw=runner.predict(r['audio_path'],cfg['evaluation']['prompt'],max_new_tokens=cfg['evaluation']['max_new_tokens']);z=normalize_label(raw,labels);out.append({'event_id':r['event_id'],'target':r['label'],'raw_prediction':raw,'prediction':z or '','correct':str(z==r['label']).lower()});a.output.parent.mkdir(parents=True,exist_ok=True);tmp=a.output.with_suffix('.tmp');
  with tmp.open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=out[0]);w.writeheader();w.writerows(out)
  tmp.replace(a.output)
  if i==1 or i%25==0 or i==len(rows):print(f'[{i}/{len(rows)}] {raw!r}',flush=True)
if __name__=='__main__':main()
