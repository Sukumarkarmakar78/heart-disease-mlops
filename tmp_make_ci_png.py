from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

folder = Path('screenshots/ci_cd')
folder.mkdir(parents=True, exist_ok=True)
texts = {
    'ci_pipeline.png': 'CI Pipeline\nLint / Test / Build / Deploy',
    'deployment_rollout.png': 'Deployment Rollout\nBuild -> Push -> Deploy -> Verify',
    'prometheus_dashboard.png': 'Prometheus Dashboard\nMetrics and Alerts',
}
for filename, text in texts.items():
    img = Image.new('RGB', (900, 600), '#ffffff')
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype('arial.ttf', 32)
    except IOError:
        font = ImageFont.load_default()
    lines = text.split('\n')
    y = 150
    for line in lines:
        bbox = draw.textbbox((0, 0), line, font=font)
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        draw.text(((900 - w) // 2, y), line, fill='black', font=font)
        y += h + 10
    draw.rectangle([50, 50, 850, 550], outline='black', width=4)
    path = folder / filename
    img.save(path)
    print('created', path)
