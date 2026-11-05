"""Bounded resumable GPU-only transport owner. No SUMO/collector dispatch."""
import hashlib,json,subprocess,sys,time
from pathlib import Path
from planfix_ops import TEMP,note
HERE=Path(__file__).parent
NAMES=('planfix_ops.py','planfix_chunk_step.py','planfix_stage_gpu_verified.py')
SOURCES={name:hashlib.sha256((HERE/name).read_bytes()).hexdigest() for name in NAMES}


def run():
 failures=0
 for batch in range(400):
  if time.time()>=1793894340 or (TEMP/'PLANFIX_GPU_RESUME_STOP.json').exists():
   note('gpuA','bounded transfer owner stopped at own stop/deadline; no remote process signalled');return
  for name,sha in SOURCES.items():assert hashlib.sha256((HERE/name).read_bytes()).hexdigest()==sha
  note('gpuA','resumable batch '+str(batch+1)+'/400; consecutive failed batches '+str(failures)+'/6')
  result=subprocess.run([sys.executable,'-u',str(HERE/'planfix_chunk_step.py')],capture_output=True,timeout=180)
  text=result.stdout.decode(errors='replace');error=result.stderr.decode(errors='replace')
  print(text,error,flush=True)
  if result.returncode:
   failures+=1
   if failures>=6:
    note('gpuA','six consecutive transfer failures; bounded owner exits, preserved chunks require explicit resume');return
   delay=min(300,60*2**(failures-1));note('gpuA','batch backoff '+str(delay)+'s');time.sleep(delay);continue
  failures=0
  if 'PACKAGE_ALL_CHUNKS_VERIFIED' not in text:
   time.sleep(3);continue
  marker=TEMP/'PLANFIX_GPU_VERIFIED_STAGE_ATTEMPT.json'
  # The stage payload independently refuses any existing worker/control root.
  # Publish the local attempt marker before the single mutation; unknown remote
  # outcomes are inspect-only and cannot auto-repeat.
  with marker.open('x') as out:json.dump(dict(unix=time.time(),sources=SOURCES),out)
  stage=subprocess.run([sys.executable,'-u',str(HERE/'planfix_stage_gpu_verified.py')],capture_output=True,timeout=100)
  print(stage.stdout.decode(errors='replace'),stage.stderr.decode(errors='replace'),flush=True)
  note('gpuA','all329 small chunks + whole source SHA verified; single fresh staging outcome rc='+str(stage.returncode)+'; inspect remote gate/key receipts before further action')
  return
 note('gpuA','400-batch limit reached; partials preserved, no remote signal')


if __name__=='__main__':
 if '--start' in sys.argv:
  assert TEMP.is_dir()
  receipt=TEMP/'PLANFIX_GPU_RESUME_OWNER.json';assert not receipt.exists(),'adopt existing owner; no duplicate'
  with (TEMP/'PLANFIX_GPU_RESUME.log').open('x',encoding='utf-8') as log:
   proc=subprocess.Popen([sys.executable,'-u',str(Path(__file__))],cwd=HERE,stdin=subprocess.DEVNULL,
     stdout=log,stderr=subprocess.STDOUT,creationflags=0x08000000)
  with receipt.open('x') as out:json.dump(dict(pid=proc.pid,unix=time.time(),sources=SOURCES,scope='GPU source transfer + one guarded fresh-stage attempt only; zero SUMO'),out)
  print('GPU_RESUME_OWNER',proc.pid,flush=True)
 else:run()
