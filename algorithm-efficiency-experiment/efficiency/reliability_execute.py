"""Sequential campaign driver; scientific gates close dependent stages automatically."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
from .common import ROOT, atomic_json, utc
from .refine_governor import GOVERNOR
from .reliability_campaign import dependency
from .reliability_spec import CAPS

def main():
    parser=argparse.ArgumentParser();parser.add_argument('campaign',type=Path);parser.add_argument('--source-campaign',type=Path,required=True)
    args=parser.parse_args();campaign=args.campaign.resolve();source=args.source_campaign.resolve()
    if (campaign/'checksums.json').exists() or (campaign/'execution.json').exists():raise ValueError('Driver requires a fresh campaign execution')
    original=GOVERNOR.read_text().strip();records=[]
    for stage in CAPS:
        if GOVERNOR.read_text().strip()!=original:raise RuntimeError('Governor not restored; halt campaign')
        ledger=json.loads((campaign/'campaign.json').read_text()) if (campaign/'campaign.json').exists() else dict(attempts=[])
        try:dependency(ledger,stage)
        except ValueError as exc:
            records.append(dict(stage=stage,state='closed',reason=str(exc),utc=utc()));atomic_json(campaign/'execution.json',records)
            print(stage+': closed by scientific dependency',flush=True);continue
        output=ROOT/'runs'/f'{campaign.name}-{stage}'
        if output.exists():raise ValueError('Run output already exists')
        argv=[sys.executable,'-B',str(ROOT/'experiment.py'),'reliability','--stage',stage,'--source-campaign',str(source),'--campaign',str(campaign),'--output',str(output)]
        record=dict(stage=stage,state='running',output=str(output),started_utc=utc(),argv=argv)
        records.append(record);atomic_json(campaign/'execution.json',records);print(stage+': starting',flush=True)
        with (campaign/(stage+'-console.log')).open('w') as log:
            result=subprocess.run(argv,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
        record.update(state='finished',returncode=result.returncode,finished_utc=utc())
        if (output/'summary.json').exists():
            summary=json.loads((output/'summary.json').read_text());audit=json.loads((output/'validation.json').read_text())
            record.update(decision=summary['decision'],audit_passed=audit['passed'])
        atomic_json(campaign/'execution.json',records);print(stage+': '+record.get('decision','startup failure'),flush=True)
        if GOVERNOR.read_text().strip()!=original:raise RuntimeError('Governor restoration failed')
        if record.get('decision') in ('candidate_selected','confirmed','no_qualified_candidate') and not record.get('audit_passed'):
            raise RuntimeError('Completed scientific stage failed its audit; stop for investigation')
    print('All eligible stages finished; final validation and campaign sealing remain.',flush=True)

if __name__=='__main__':main()
