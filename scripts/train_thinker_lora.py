#!/usr/bin/env python3
"""Train q/v-projection LoRA on fixed BEANS train split; monitor valid loss."""
from __future__ import annotations
import argparse,csv,json,random
from pathlib import Path
import numpy as np,torch,yaml
from peft import LoraConfig,get_peft_model
from animal_omni.qwen_runner import QwenThinkerRunner

def main():
 p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--manifest',type=Path,required=True);p.add_argument('--model-id',required=True);p.add_argument('--output-dir',type=Path,required=True);p.add_argument('--epochs',type=int,default=1);p.add_argument('--lr',type=float,default=2e-4);p.add_argument('--rank',type=int,default=8);p.add_argument('--accumulation',type=int,default=8);p.add_argument('--limit-train',type=int);p.add_argument('--limit-valid',type=int,default=64);a=p.parse_args()
 random.seed(20250813);np.random.seed(20250813);torch.manual_seed(20250813);cfg=yaml.safe_load(a.config.read_text());rows=list(csv.DictReader(a.manifest.open()));train=[r for r in rows if r['split']=='train'];valid=[r for r in rows if r['split']=='valid'];random.shuffle(train)
 if a.limit_train:train=train[:a.limit_train]
 if a.limit_valid:valid=valid[:a.limit_valid]
 runner=QwenThinkerRunner(a.model_id);runner.model.thinker.enable_input_require_grads();runner.model.thinker.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
 lora=LoraConfig(r=a.rank,lora_alpha=2*a.rank,lora_dropout=.05,target_modules=['q_proj','v_proj'],bias='none',task_type='CAUSAL_LM');runner.model.thinker=get_peft_model(runner.model.thinker,lora);runner.model.thinker.print_trainable_parameters();opt=torch.optim.AdamW((x for x in runner.model.thinker.parameters() if x.requires_grad),lr=a.lr)
 history=[];a.output_dir.mkdir(parents=True,exist_ok=True)
 for epoch in range(1,a.epochs+1):
  runner.model.thinker.train();opt.zero_grad(set_to_none=True);losses=[]
  for i,r in enumerate(train,1):
   inputs=runner.teacher_forced_inputs(r['audio_path'],cfg['evaluation']['prompt'],r['label']);loss=runner.model.thinker(**inputs,use_cache=False,return_dict=True).loss/a.accumulation;loss.backward();losses.append(float(loss.detach())*a.accumulation)
   if i%a.accumulation==0 or i==len(train):torch.nn.utils.clip_grad_norm_(runner.model.thinker.parameters(),1.);opt.step();opt.zero_grad(set_to_none=True)
   if i==1 or i%25==0 or i==len(train):print(f'epoch={epoch} [{i}/{len(train)}] loss={np.mean(losses[-25:]):.4f}',flush=True)
  runner.model.thinker.eval();vl=[]
  with torch.inference_mode():
   for r in valid:
    inputs=runner.teacher_forced_inputs(r['audio_path'],cfg['evaluation']['prompt'],r['label']);vl.append(float(runner.model.thinker(**inputs,use_cache=False,return_dict=True).loss))
  rec={'epoch':epoch,'train_loss':float(np.mean(losses)),'valid_loss':float(np.mean(vl))};history.append(rec);print(rec,flush=True);runner.model.thinker.save_pretrained(a.output_dir/f'epoch_{epoch}');(a.output_dir/'history.json').write_text(json.dumps(history,indent=2))
if __name__=='__main__':main()
