# make_video py title
import os, sys, subprocess, tempfile
import matplotlib
from PIL import Image, ImageDraw, ImageFont

src, out = sys.argv[1], sys.argv[2]
T = os.path.join(os.path.dirname(matplotlib.__file__), "mpl-data", "fonts", "ttf")
BOLD, REG = os.path.join(T, "DejaVuSans-Bold.ttf"), os.path.join(T, "DejaVuSans.ttf")
W, H = 1280, 860
tmp = tempfile.mkdtemp()


def card(path, lines):
    im = Image.new("RGB", (W, H), "#fcfcfb"); d = ImageDraw.Draw(im)
    y = H // 2 - sum(s + 18 for _, s, _ in lines) // 2
    for text, size, col in lines:
        f = ImageFont.truetype(BOLD if size >= 30 else REG, size)
        w = d.textlength(text, font=f)
        d.text(((W - w) / 2, y), text, font=f, fill=col); y += size + 18
    im.save(path)


card(os.path.join(tmp, "title.png"), [
    ("Polarization Pipeline", 54, "#0b0b0b"),
    ("Measuring political polarization in 1.75M election tweets", 26, "#52514e"),
    ("Live demo: one tweet through every version I built", 24, "#2a78d6"),
    ("Adrian Polishchuk Carrillo", 20, "#8a8984")])
card(os.path.join(tmp, "end.png"), [
    ("Held-out accuracy: 8% (Mark 1) → 76% (Mark 4) → 80% (final)", 30, "#0b0b0b"),
    ("Code + full commit history: github.com/adripoli/Polarization-on-Social-Media", 22, "#52514e"),
    ("Run it yourself:  python demo/app.py", 22, "#2a78d6")])


def run(*a):
    subprocess.run(["ffmpeg", "-v", "error", "-y", *a], check=True)


enc = ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-r", "25", "-vf", f"scale={W}:{H}"]
run("-loop", "1", "-t", "3.5", "-i", os.path.join(tmp, "title.png"), *enc, os.path.join(tmp, "a.mp4"))
run("-i", src, *enc, os.path.join(tmp, "b.mp4"))
run("-loop", "1", "-t", "4", "-i", os.path.join(tmp, "end.png"), *enc, os.path.join(tmp, "c.mp4"))
lst = os.path.join(tmp, "list.txt")
open(lst, "w").write("".join(f"file '{os.path.join(tmp, x)}'\n".replace("\\", "/") for x in ("a.mp4", "b.mp4", "c.mp4")))
run("-f", "concat", "-safe", "0", "-i", lst, "-c", "copy", "-movflags", "+faststart", out)
print("wrote", out)
