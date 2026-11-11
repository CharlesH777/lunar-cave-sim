"""Join showcase and sensor playback into a verified, browser-playable H.264 demo."""
import argparse,hashlib,json,os,subprocess
from pathlib import Path

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('directory',type=Path)
a=p.parse_args();folder=a.directory.resolve()
temporary=folder/f'lunar_cave_demo.encoding_{os.getpid()}.mp4'
sources=[folder/'cave_showcase.mp4',folder/'search_episode.mp4']
if not all(x.is_file() for x in sources):raise SystemExit('Both showcase and search videos are required')
command=['ffmpeg','-hide_banner','-loglevel','error','-y','-i',str(sources[0]),'-i',str(sources[1]),
         '-filter_complex_threads','2','-filter_complex',
         '[0:v]fps=24,setsar=1[v0];[1:v]fps=24,scale=1280:720,setsar=1[v1];[v0][v1]concat=n=2:v=1:a=0[v]',
         '-map','[v]','-r','24','-c:v','libx264','-threads','6','-preset','fast','-crf','20',
         '-pix_fmt','yuv420p','-movflags','+faststart',str(temporary)]
with (folder/'video_encoding.log').open('w') as log:
    subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True)
verification=subprocess.run(['ffmpeg','-hide_banner','-v','error','-i',str(temporary),'-f','null','-'],
                            stdout=subprocess.DEVNULL,stderr=subprocess.PIPE,check=True)
if verification.stderr:raise RuntimeError(verification.stderr.decode()[:500])
meta=json.loads(subprocess.check_output(['ffprobe','-v','fatal','-select_streams','v:0','-show_entries',
    'stream=codec_name,width,height,r_frame_rate,nb_frames:format=duration,size','-of','json',str(temporary)]))
assert meta['streams'][0]['codec_name']=='h264'
assert meta['streams'][0]['r_frame_rate']=='24/1'
temporary.replace(folder/'lunar_cave_demo.mp4')
meta.update(full_decode_passed=True,sha256=hashlib.sha256((folder/'lunar_cave_demo.mp4').read_bytes()).hexdigest())
(folder/'video_validation.json').write_text(json.dumps(meta,indent=2)+'\n')
print(json.dumps({'video':str(folder/'lunar_cave_demo.mp4'),'seconds':meta['format']['duration'],'passed':True}))
