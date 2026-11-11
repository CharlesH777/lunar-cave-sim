"""Build the eight authored layouts sequentially on one GPU; preserve Earth."""
import argparse,json,subprocess,time
from pathlib import Path

ROOT=Path(__file__).resolve().parent
p=argparse.ArgumentParser();p.add_argument('--maps',nargs='+',default=[f'{i:02d}' for i in range(1,9)])
p.add_argument('--output-root',type=Path)
p.add_argument('--planner',choices=['astar','hybrid'],default='hybrid')
p.add_argument('--scene-root',type=Path,default=ROOT/'outputs/maps_revised')
p.add_argument('--previews-only',action='store_true');args=p.parse_args()
folder=(args.output_root or ROOT/'outputs'/('maps_smooth_tuned' if args.planner=='hybrid' else 'maps_astar')).resolve();folder.mkdir(parents=True,exist_ok=True)
previous=json.loads((folder/'build_status.json').read_text()) if (folder/'build_status.json').exists() else {}
report={'phase':'building','maps':previous.get('maps',{}),'started':time.strftime('%Y-%m-%dT%H:%M:%S%z')}
def save():
    tmp=folder/'build_status.tmp';tmp.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n');tmp.replace(folder/'build_status.json')
for map_id in args.maps:
    if (folder/'REQUEST_STOP').exists():report['phase']='stopped';save();break
    out=folder/('map_'+map_id);out.mkdir(exist_ok=True)
    report['current_map']=map_id;save()
    command=['bash',str(ROOT/'run_demo.sh'),'--map',map_id,'--seed',str(70+int(map_id)),'--output',str(out),'--planner',args.planner]
    source=args.scene_root/('map_'+map_id)
    if (source/'cave.usdc').exists():command+=['--scene-dir',str(source)]
    if map_id=='01' and not args.previews_only:command+=['--video-frames','360']
    else:command+=['--preview-only']
    print('BUILD_MAP',map_id,flush=True)
    start=time.time()
    with (out/'run.log').open('w') as log:
        proc=subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
    state=json.loads((out/'status.json').read_text()) if (out/'status.json').exists() else {'phase':'missing_status'}
    ok=state['phase'] in ['complete','preview_complete']
    if map_id=='01' and not args.previews_only:
        result=out/'episode_000/result.json'
        ok=ok and result.exists() and json.loads(result.read_text()).get('success') is True
    if not (out/'cave.usdc').exists() or not (out/'hero.png').exists():ok=False
    report['maps'][map_id]={'ok':ok,'phase':state['phase'],'seconds':round(time.time()-start,2),'output':str(out)};save()
    print('MAP_RESULT',map_id,report['maps'][map_id],flush=True)
    if not ok:report['phase']='failed';save();raise SystemExit(1)
else:
    report['phase']='complete';report['finished']=time.strftime('%Y-%m-%dT%H:%M:%S%z');save()
    print('EIGHT_MAPS_COMPLETE',flush=True)
