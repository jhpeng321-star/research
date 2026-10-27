"""Own detached spot notice + observable collection mirror; never signals jobs.

Only cycle>=2 main collector output is mirrored. Source/route/assignment files
and restricted paths are excluded before rsync traversal. Metadata requests do
not start SUMO. Completion artifacts remain authoritative, partial copies are
recovery evidence only and are never adopted without HASHES validation.
"""
import argparse
import json
import os
from pathlib import Path
import time
import urllib.request
import urllib.error

def notice():
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    # cpuP/cpuQ are Tencent CVM spot instances. 404 means no scheduled reclaim.
    try:
        with opener.open('http://metadata.tencentyun.com/latest/meta-data/spot/termination-time', timeout=2) as response:
            return {'reclaim': True, 'provider': 'tencent', 'notice': response.read().decode()}
    except urllib.error.HTTPError as error:
        if error.code == 404:
            return {'reclaim': False, 'provider': 'tencent', 'metadata_status': '404 no termination scheduled'}
        tencent_error = str(error)
    except Exception as error:
        tencent_error = str(error)
    base = 'http://169.254.169.254/latest/'
    try:
        request = urllib.request.Request(base + 'api/token', method='PUT',
            headers={'X-aws-ec2-metadata-token-ttl-seconds': '60'})
        with opener.open(request, timeout=2) as response:
            token = response.read().decode()
        request = urllib.request.Request(base + 'meta-data/spot/instance-action',
            headers={'X-aws-ec2-metadata-token': token})
        with opener.open(request, timeout=2) as response:
            return {'reclaim': True, 'notice': response.read().decode()}
    except Exception as error:
        return {'reclaim': False, 'metadata_status': str(error), 'tencent_status': tencent_error,
                'notice_available': False}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base', required=True)
    parser.add_argument('--box', required=True, choices=['cpuP', 'cpuQ'])
    parser.add_argument('--once', action='store_true')
    args = parser.parse_args()
    base = Path(args.base)
    own = base / 'autonomy'
    while True:
        row = {'unix': time.time(), 'pid': os.getpid(), 'sid': os.getsid(0), 'box': args.box,
               'notice': notice(), 'mirrors': []}
        # gpuA pulls through its existing authenticated collector connections. The
        # collectors deliberately need no gpuA private key or laptop SSH alias.
        row['mirror_mode'] = 'gpuA_detached_pull'
        row['observable_sources'] = ['cycle>=2 artifact', 'cycle>=2 collect/branches']
        temp = own / 'SPOT_STATUS.tmp'
        temp.write_text(json.dumps(row, indent=2))
        temp.replace(own / 'SPOT_STATUS.json')
        if args.once:
            print(json.dumps(row))
            return
        time.sleep(10 if row['notice']['reclaim'] else 30)


if __name__ == '__main__':
    main()
