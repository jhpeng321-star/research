"""Reuse native PPO residual network; MAPPO anchors injected before original safety."""
import math
from tools.coopnative.policy import Policy, encode as local_encode, targets
from tools.coopfree.policy import leader_gap, requests, TRACK_S


def encode(rows, commands):
    context=local_encode(rows)
    byid={r['vid']:r for r in rows}
    anchors=[]
    for vid in context['members']:
        own=byid[vid]; speed=max(0.,own['v_mps']+TRACK_S*commands[vid])
        gap=leader_gap(own,[r for r in rows if abs(r['s_m']-own['s_m'])<=100])
        tau=1. if gap is None else max(1e-6,gap/max(speed,1e-6))
        anchors.append([speed,math.log(tau)])
    context['anchors']=anchors
    return context


def group(rows):
    return [r['vid'] for r in sorted(rows,key=lambda r:(abs(r['s_m']),r['vid']))
            if r['is_cav'] and -100<=r['s_m']<=75.44][:2]


class Controller:
    def __init__(self, policy=None, generator=None, deterministic=False, samples=None,
                 branch=None, guide=None, beta=0.):
        self.policy,self.generator,self.deterministic=policy,generator,deterministic
        self.samples,self.branch=samples,branch
        self.events=[]; self.targets={}; self.until=None
        self.guide,self.beta=guide,beta

    def apply(self, rows, commands, now):
        if self.until is None or now>=self.until:
            if self.samples is not None:
                sample=self.samples.get(str(int(now)))
                if sample is None:
                    self.targets={}; self.until=now+10
                    return {}
            elif self.policy is None:
                self.targets={}; self.until=now+10
                return {}
            else:
                sample=self.policy.sample(encode(rows,commands),self.generator,
                                          deterministic=self.deterministic)
            proposal=None
            if self.guide is not None and self.samples is None and not self.deterministic:
                from .exploration import route,apply_proposal
                if route(sample,rows,self.beta,self.generator):
                    proposal=self.guide(rows,commands,now,sample)
                    sample=apply_proposal(sample,proposal)
            self.targets=targets(sample); self.until=now+sample['window']
            if self.branch and now>=self.branch['time_s']:
                if now==self.branch['time_s']:
                    self.targets={v:t for v,t in self.targets.items() if v not in self.branch['revert']}
                else:
                    self.targets={} # All twins release to MAPPO after intervention window.
            self.events.append(dict(time_s=int(now),sample=sample,rows=rows,
                                    targets=dict(self.targets),group=group(rows),proposal=proposal))
        return requests(self.targets,rows,zone=None) if now<self.until else {}
