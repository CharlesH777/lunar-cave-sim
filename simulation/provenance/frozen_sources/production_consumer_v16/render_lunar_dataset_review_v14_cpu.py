"""Render actual prompt/CoT sheets and native-frame five-times video previews."""
import argparse
import json
import math
from pathlib import Path
import subprocess
from PIL import Image, ImageDraw, ImageFont


def records(path):
    with path.open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def choose_font(argument, size):
    paths = [Path(argument)] if argument else [
        Path('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'),
        Path('/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc')]
    for path in paths:
        if path.is_file():
            return ImageFont.truetype(str(path), size)
    raise FileNotFoundError('Set --font to a font supporting the dataset language')


def wrap(text, font, width):
    lines, current = [], ''
    for char in text:
        if current and font.getlength(current + char) > width:
            lines.append(current)
            current = char
        else:
            current += char
    if current:
        lines.append(current)
    return lines


def render(dataset, output, *, font_path=None, video=True):
    dataset = Path(dataset).resolve(strict=True)
    output = Path(output).absolute()
    output.mkdir(parents=True, exist_ok=False)
    manifest = json.loads((dataset / 'dataset_manifest.json').read_text())
    font, small = choose_font(font_path, 16), choose_font(font_path, 13)
    review = ['# 实际数据集预览', '',
              '每张图为原始 RGB224；面板不裁切、不放大。视频为原始顺序帧五倍速，无插帧。', '']
    bindings = []
    input_variants = []
    for entry in manifest['episodes']:
        name = entry['episode_id']
        trajectory, language = dataset / entry['trajectory'], dataset / entry['language']
        frames = {x['step_index']: x for x in records(trajectory / 'frame_records.jsonl')}
        labels = [x for x in records(language / 'annotations.jsonl') if x.get('cot_valid')]
        queries = json.loads((language / 'queries.json').read_text())
        folder = output / name
        folder.mkdir()
        selected_query = next((row for row in queries['variants'] if row.get('query_partition') == 'train'), None)
        instruction = selected_query['instruction'] if selected_query else queries['canonical_instruction']
        input_variants.append(dict(episode_id=name, instruction=instruction,
            language_episode_id=selected_query.get('language_episode_id') if selected_query else None,
            family_id=selected_query.get('family_id') if selected_query else None,
            query_partition=selected_query.get('query_partition') if selected_query else 'canonical'))
        header = wrap('输入指令（实际训练 query）：' + instruction, font, 4 * 224 + 3 * 14)
        top = 34 + len(header) * 23
        rows = max(1, math.ceil(len(labels) / 4))
        canvas = Image.new('RGB', (4 * 224 + 5 * 14, top + rows * 345 + 14), '#14232b')
        draw = ImageDraw.Draw(canvas)
        for n, line in enumerate(header):
            draw.text((14, 12 + n * 23), line, font=font, fill='white')
        for n, annotation in enumerate(labels):
            step = annotation['step_index']
            frame = frames[step]
            x, y = 14 + (n % 4) * 238, top + (n // 4) * 345
            image_path = trajectory / frame['rgb_224']
            with Image.open(image_path) as source:
                if source.size != (224, 224):
                    raise ValueError('Review requires the exact RGB224 observation')
                canvas.paste(source.convert('RGB'), (x, y))
            stamp = frame['inputs']['observation']['timestamp_s']
            draw.text((x, y + 230), f'第 {step} 帧 · {stamp:.1f}s', font=small, fill='#bed1dd')
            for line_index, line in enumerate(wrap(annotation['cot_text'], font, 224)):
                draw.text((x, y + 252 + line_index * 22), line, font=font, fill='white')
            bindings.append(dict(episode_id=name, step_index=step, timestamp_s=stamp,
                                 rgb224=str(image_path.relative_to(dataset)), cot_text=annotation['cot_text']))
        canvas.save(folder / 'prompt_cot.jpg', quality=95, subsampling=0)
        review += ['## ' + name, '', '> ' + instruction, '',
                   f'[提示词与 CoT 图文]({name}/prompt_cot.jpg)', '',
                   '| 家族 | 分区 | query |', '|---|---|---|']
        for query in queries['variants']:
            family = query.get('family_id', '')
            partition = query.get('query_partition', query.get('partition', ''))
            review.append('| ' + family + ' | ' + partition + ' | ' + query['instruction'].replace('|', '／') + ' |')
        if video:
            subprocess.run(['ffmpeg', '-nostdin', '-hide_banner', '-loglevel', 'error', '-n',
                '-framerate', '25', '-start_number', '0', '-i', str(trajectory / 'frames/%06d_rgb224.png'),
                '-frames:v', str(len(frames)), '-an', '-c:v', 'libx264', '-crf', '18', '-pix_fmt', 'yuv420p',
                str(folder / 'preview_5x.mp4')], check=True)
            review += ['', f'[整条路线五倍速]({name}/preview_5x.mp4)', '']
    (output / 'README.md').write_text('\n'.join(review) + '\n')
    (output / 'bindings.json').write_text(json.dumps(dict(native_rgb224=True, resized=False,
        interpolated_frames=False, input_variants=input_variants, bindings=bindings), ensure_ascii=False, indent=2) + '\n')
    return dict(passed=True, episodes=len(manifest['episodes']), cot_frames=len(bindings), output=str(output))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--font', type=Path)
    parser.add_argument('--no-video', action='store_true')
    args = parser.parse_args()
    print(json.dumps(render(args.dataset, args.output, font_path=args.font,
                            video=not args.no_video), ensure_ascii=False))


if __name__ == '__main__':
    main()
