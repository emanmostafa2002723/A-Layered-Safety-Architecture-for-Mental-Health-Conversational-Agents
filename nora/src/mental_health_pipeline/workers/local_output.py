"""Isolated project-owned local output judge; no provider approval is invented."""
import argparse
from contextlib import redirect_stdout
from pathlib import Path
import json
import sys
from mental_health_pipeline.workers.common import announce_ready,announce_startup_error,_send


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--component-root',type=Path,required=True)
    parser.add_argument('--settings',type=Path,required=True)
    parser.add_argument('--hybrid-project-root',type=Path)
    args=parser.parse_args()
    try:
        with redirect_stdout(sys.stderr):
            from mental_health_pipeline.local_output import LocalJudge,JudgeSettings
            if args.hybrid_project_root:
                from mental_health_pipeline.hybrid_output import HybridJudge
                judge=HybridJudge.load(args.component_root,args.settings,args.hybrid_project_root)
            else:
                judge=LocalJudge(args.component_root,JudgeSettings.load(args.settings))
    except Exception:
        announce_startup_error('output');return 2
    announce_ready('output')
    while True:
        line=sys.stdin.buffer.readline(262145)
        if not line:return 0
        request_id='unknown'
        try:
            if len(line)>262144 or not line.endswith(b'\n'):return 2
            request=json.loads(line);request_id=request.get('request_id')
            if request.get('schema_version')!='worker-request/1.0' or not isinstance(request_id,str) or not request_id:
                raise ValueError('Invalid envelope')
            if request.get('operation')=='shutdown':return 0
            if request.get('operation')!='assess' or set(request['payload'])!={'input'}:
                raise ValueError('Invalid operation')
            with redirect_stdout(sys.stderr):result=judge.assess(request['payload']['input'])
            _send({'schema_version':'worker-response/1.0','request_id':request_id,'status':'ok','result':result})
        except Exception:
            _send({'schema_version':'worker-response/1.0','request_id':request_id,
                   'status':'error','error_code':'local_output_request_failed'})


if __name__=='__main__':raise SystemExit(main())
