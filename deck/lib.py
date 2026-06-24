"""Deck build library — shared design tokens + components for the poverty-cnn deck.
Cinematic earth-observation language (see DESIGN-SYSTEM.md). Render via headless Chrome.

Compose a slide:  body = header(...) + logic_blocks(...) + ... ; write_slide("s04", "dark", body, css="...")
Then render_all() and Read the PNGs to self-verify.
"""
import subprocess, pathlib
DECK = pathlib.Path(__file__).parent
SL = DECK / "slides"; SL.mkdir(exist_ok=True)
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"

FONTS = ('<link rel=preconnect href=https://fonts.googleapis.com>'
 '<link href="https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:wght@600;700;800'
 '&family=Hanken+Grotesk:wght@400;500;600;700&family=JetBrains+Mono:wght@500&display=swap" rel=stylesheet>')
GRAIN = ("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='180' height='180'%3E"
 "%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.85' numOctaves='2'/%3E"
 "%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23n)'/%3E%3C/svg%3E")
GLOBE = (DECK / "assets/_globe.svg").read_text()

DARK_BG = ("background:radial-gradient(58% 75% at 80% 26%, rgba(24,180,205,.26), transparent 60%),"
 "radial-gradient(54% 64% at 16% 88%, rgba(98,72,225,.22), transparent 62%),"
 "radial-gradient(38% 46% at 96% 92%, rgba(222,58,140,.12), transparent 60%),"
 "linear-gradient(170deg,#060e1a,#040a13 60%,#03070e);")
LIGHT_BG = ("background:radial-gradient(46% 56% at 92% 6%, rgba(24,180,205,.11), transparent 60%),"
 "radial-gradient(40% 50% at 2% 99%, rgba(98,72,225,.06), transparent 60%),#eef1f4;")

# shared classes used across slides (both themes; colour overridden by theme block)
COMMON = """
*{margin:0;padding:0;box-sizing:border-box}
html,body{width:1920px;height:1080px;overflow:hidden}
body{position:relative;font-family:'Hanken Grotesk',sans-serif}
.grain{position:absolute;inset:0;background-image:url("GRAINURL");background-size:200px;pointer-events:none}
.svg{position:absolute;inset:0;z-index:0}
.pg{position:absolute;top:50px;left:118px;font-family:'JetBrains Mono';font-size:16px;letter-spacing:3px;z-index:6}
.eb{position:absolute;top:50px;right:118px;font-family:'JetBrains Mono';font-size:16px;letter-spacing:4px;z-index:6}
.head{position:absolute;top:96px;left:118px;width:1690px;z-index:5}
.h1{font-family:'Bricolage Grotesque';font-weight:800;font-size:58px;letter-spacing:-1px;line-height:1.0}
.h2{font-size:25px;margin-top:12px;font-weight:500;max-width:1500px}
.blk{position:relative;padding-left:22px;margin-bottom:24px}
.blk:before{content:'';position:absolute;left:0;top:3px;width:5px;height:calc(100% - 6px);border-radius:3px;background:linear-gradient(180deg,#18cfe6,#1fae8b)}
.bl{font-family:'JetBrains Mono';font-size:14px;letter-spacing:2px;color:#1ec7da;font-weight:500;margin-bottom:6px}
.bt{font-size:22.5px;line-height:1.48}
.strip{position:absolute;left:118px;bottom:52px;display:flex;gap:16px;z-index:5}
.stat{border-radius:13px;padding:15px 24px;min-width:236px}
.stat .n{font-family:'Bricolage Grotesque';font-weight:800;font-size:30px}
.stat .c{font-size:15px;margin-top:3px}
.method{position:absolute;left:118px;bottom:48px;width:1684px;background:#eef7f9;border:1px solid #cfe6ec;border-left:5px solid #16b9d0;border-radius:12px;padding:17px 26px;z-index:5;display:flex;align-items:baseline;gap:22px}
.method .mtag{font-family:'JetBrains Mono';font-size:13px;letter-spacing:2px;color:#0e8ba0;font-weight:600;white-space:nowrap}
.method .mtxt{font-size:20px;color:#3a4c56;line-height:1.42}.method .mtxt b{color:#0f2636}
.verdict{position:absolute;left:118px;bottom:168px;width:1684px;background:#0f2636;border-radius:14px;padding:22px 34px;z-index:5;display:flex;align-items:center;gap:22px}
.verdict .va{font-family:'Bricolage Grotesque';font-weight:800;font-size:36px;color:#3fe3f0;flex:none;line-height:1}
.verdict .vt{font-family:'Bricolage Grotesque';font-weight:700;font-size:27px;color:#fff;line-height:1.26}.verdict .vt b{color:#3fe3f0}
"""
DARK_THEME = """
body{color:#eaf4fa;}
.grain{opacity:.05;mix-blend-mode:overlay}
.pg{color:#5a7588}.eb{color:#46e3f0}
.h1{color:#fff}.h1 em{font-style:normal;background:linear-gradient(96deg,#7af0ff,#34d6c0);-webkit-background-clip:text;background-clip:text;color:transparent}
.h2{color:#9fb4c6}.h2 b{color:#cfe8f0;font-weight:700}
.bt{color:#b6c8d6}.bt b{color:#fff;font-weight:700}.bt .hot{color:#46e3f0;font-weight:700}
.stat{background:rgba(11,34,51,.62);border:1px solid rgba(70,227,240,.25)}.stat .n{color:#3fe3f0}.stat .c{color:#a9bcc8}
"""
LIGHT_THEME = """
body{color:#16303f;}
.grain{opacity:.18;mix-blend-mode:multiply}
.pg{color:#8aa0ad}.eb{color:#1296ad}
.h1{color:#0f2636}.h1 em{font-style:normal;color:#1296ad}
.h2{color:#566b78}.h2 b{color:#15303f;font-weight:700}
.bt{color:#33454f}.bt b{color:#0f2636;font-weight:700}.bt .hot{color:#1296ad;font-weight:700}
.bl{color:#1296ad}
.stat{background:#0f2636}.stat .n{color:#3fe3f0}.stat .c{color:#a9bcc8}
"""

def header(title, sub="", eyebrow="", pageno="", color_h1=None):
    h = f'<div class=pg>{pageno}</div><div class=eb>{eyebrow}</div><div class=head>'
    h += f'<div class=h1>{title}</div>'
    if sub: h += f'<div class=h2>{sub}</div>'
    return h + '</div>'

def logic_blocks(items, left, top, width):
    """items: list of (LABEL, html_text)."""
    s = f'<div style="position:absolute;left:{left}px;top:{top}px;width:{width}px;z-index:5">'
    for lab, txt in items:
        s += f'<div class=blk><div class=bl>{lab}</div><div class=bt>{txt}</div></div>'
    return s + '</div>'

def stat_chips(items):
    """items: list of (number, caption)."""
    s = '<div class=strip>'
    for n, c in items:
        s += f'<div class=stat><div class=n>{n}</div><div class=c>{c}</div></div>'
    return s + '</div>'

def globe():
    return f'<div class=svg>{GLOBE}</div>'

def method_note(text):
    """A 'HOW WE GOT THIS' process ribbon for results slides."""
    return f'<div class=method><div class=mtag>HOW WE GOT THIS</div><div class=mtxt>{text}</div></div>'

def verdict(text):
    """A bold one-line takeaway banner (dark) for the message of a slide."""
    return f'<div class=verdict><div class=va>&rarr;</div><div class=vt>{text}</div></div>'

def write_slide(name, theme, body, css=""):
    bg = DARK_BG if theme == "dark" else LIGHT_BG
    theme_css = DARK_THEME if theme == "dark" else LIGHT_THEME
    full = (f"<!doctype html><html><head><meta charset=utf-8>{FONTS}<style>"
            f"{COMMON.replace('GRAINURL', GRAIN)}\nbody{{{bg}}}\n{theme_css}\n{css}"
            f"</style></head><body><div class=grain></div>{body}</body></html>")
    (SL / f"{name}.html").write_text(full)
    return SL / f"{name}.html"

def render(name):
    out = SL / f"{name}.png"
    subprocess.run([CHROME, "--headless=new", "--hide-scrollbars", "--force-device-scale-factor=1",
        "--window-size=1920,1080", "--virtual-time-budget=4500",
        f"--screenshot={out}", f"file://{SL/(name+'.html')}"],
        stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL)
    return out
