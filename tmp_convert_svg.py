from pathlib import Path
from cairosvg import svg2png
src_dir = Path('screenshots/ci_cd')
for name in [
    'ci_pipeline.svg',
    'deployment_rollout.svg',
    'prometheus_dashboard.svg',
]:
    svg_path = src_dir / name
    png_path = src_dir / (svg_path.stem + '.png')
    svg2png(url=str(svg_path), write_to=str(png_path), output_width=900)
    print('converted', png_path)
