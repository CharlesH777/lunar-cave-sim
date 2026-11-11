"""Build one fresh morphology-focused representative cavity; no Kit or GPU."""
import argparse
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import time
import traceback


def write(path,value):
    path=Path(path);tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n');tmp.replace(path)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=Path(__file__).parent/'outputs/lunar_morphology_v4/representative_01')
    p.add_argument('--spacing',type=float,default=.28)
    args=p.parse_args();output=args.output.absolute();output.mkdir(parents=True,exist_ok=False)
    start=time.monotonic()
    def status(phase,**kw):
        value=dict(phase=phase,updated_utc=datetime.now(timezone.utc).isoformat(),elapsed_seconds=time.monotonic()-start,
                   GPU_work=False,simulation_started=False,training_started=False,**kw)
        write(output/'status.json',value);print(json.dumps(value),flush=True)
    try:
        if os.environ.get('CUDA_VISIBLE_DEVICES') not in (None,'','-1'):
            raise RuntimeError("Set CUDA_VISIBLE_DEVICES='' for CPU authoring")
        import numpy as np
        from cave.lunar_morphology_v4 import candidate_config,build_geometry,recommended_views,floor_height_at
        from cave.network3d_scene import export_network_usd
        config=candidate_config();config['recommended_views']=recommended_views(config)
        xy=config['recommended_scale_rover_xy_m']
        config['recommended_scale_rover_ground_xyz_m']=[*xy,float(floor_height_at(config,*xy))]
        write(output/'morphology.json',config)
        status('building_cpu_geometry')
        geometry=build_geometry(config,args.spacing)
        if not geometry['report']['passed']:
            write(output/'geometry_report.json',geometry['report']);status('rejected',usd_authored=False);return 2
        status('exporting_usd',vertices=len(geometry['vertices']),triangles=len(geometry['faces']))
        export=export_network_usd(config,geometry,output/'cave.usdc')
        np.savez_compressed(output/'mesh.npz',vertices=export['vertices'],faces=export['faces'])
        report=geometry['report'];report['usd_export']=export['report']
        write(output/'geometry_report.json',report)
        status('complete',passed=True,usd='cave.usdc');return 0
    except BaseException as exc:
        status('failed',error=repr(exc));write(output/'error.json',dict(error=repr(exc),traceback=traceback.format_exc()));raise


if __name__=='__main__':raise SystemExit(main())
