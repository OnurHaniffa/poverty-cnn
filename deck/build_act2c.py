"""Act 2 finish: S9 the 8 bands (meaningful swatches) + S10 the model."""
import base64, pathlib
from lib import write_slide, render, header
DECK = pathlib.Path(__file__).parent
def img(p):
    return "data:image/png;base64," + base64.b64encode((DECK/"assets/bands"/p).read_bytes()).decode()
def blk(label, text):
    return f'<div class=blk><div class=bl>{label}</div><div class=bt>{text}</div></div>'

RGB=img("rgb.png"); NIR=img("nir.png"); SWIR=img("swir.png"); TH=img("thermal.png"); NL=img("nl.png")

# ---------- S9 · the 8 bands ----------
cards = [
 (RGB,"Colour","R · G · B","What your eye would see &mdash; roofs, roads, the layout of the settlement.",False),
 (NIR,"Near-infrared","NIR","Vegetation vigour and rooftop materials the eye can&rsquo;t see.",False),
 (SWIR,"Shortwave IR","SWIR 1·2","Built-up surfaces, bare soil, and moisture.",False),
 (TH,"Thermal","TEMP","Surface heat &mdash; denser, built-up areas read hotter.",False),
 (NL,"Night-lights","NL","Electrification at night &mdash; the <b>strongest</b> single wealth cue.",True),
]
s9_css = """
.samecap{position:absolute;left:118px;top:250px;font-family:'JetBrains Mono';font-size:15px;letter-spacing:2px;color:#0e8ba0;z-index:5}
.samecap b{color:#0f2636}
.bands{position:absolute;left:118px;top:292px;width:1684px;display:flex;gap:20px;z-index:5}
.bc{flex:1;background:#fff;border:1px solid #dbe5ea;border-radius:16px;overflow:hidden;box-shadow:0 5px 16px rgba(20,40,55,.05);position:relative}
.bc.key{border:2px solid #16b9d0;box-shadow:0 10px 24px rgba(22,185,208,.22)}
.bc img{width:100%;height:188px;object-fit:cover;display:block}
.tag{position:absolute;top:12px;right:12px;background:#16b9d0;color:#fff;font-family:'JetBrains Mono';font-size:11px;letter-spacing:1px;padding:5px 9px;border-radius:6px}
.body{padding:17px 19px 20px}
.bn{font-family:'Bricolage Grotesque';font-weight:800;font-size:23px;color:#0f2636}
.bb{font-family:'JetBrains Mono';font-size:12px;color:#1296ad;letter-spacing:1px;margin:3px 0 10px}
.bm{font-size:15.5px;color:#566b78;line-height:1.4}.bm b{color:#0f2636}
.bot{position:absolute;left:118px;top:728px;width:1684px;display:flex;gap:30px;z-index:5}
.bcard{flex:1;background:#fff;border:1px solid #dbe5ea;border-left:5px solid #16b9d0;border-radius:16px;padding:26px 32px;box-shadow:0 5px 16px rgba(20,40,55,.05)}
.bcl{font-family:'JetBrains Mono';font-size:14px;letter-spacing:2.5px;color:#1296ad;margin-bottom:11px}
.bct{font-size:23px;line-height:1.5;color:#33454f}.bct b{color:#0f2636;font-weight:700}
"""
s9 = header("Eight channels &mdash; <em>eight ways to see a village</em>",
            "Not eight photos &mdash; <b>one image, eight channels deep</b> (like RGB, but with 8). Each channel exposes a different physical signature of wealth.",
            eyebrow="02 · THE INGREDIENTS", pageno="11 / 33")
s9 += '<div class=samecap>&darr;&nbsp; ALL FIVE ARE <b>THE SAME VILLAGE</b> &mdash; ONE URBAN CLUSTER IN MALI, SEEN THROUGH ITS EIGHT BANDS</div>'
s9 += '<div class=bands>'
for src,name,band,mean,key in cards:
    tag = '<div class=tag>KEY SIGNAL</div>' if key else ''
    s9 += (f'<div class="bc {"key" if key else ""}"><img src="{src}">{tag}'
           f'<div class=body><div class=bn>{name}</div><div class=bb>{band}</div><div class=bm>{mean}</div></div></div>')
s9 += '</div>'
s9 += '<div class=bot>'
s9 += ('<div class=bcard><div class=bcl>HOW EACH CHANNEL IS MADE</div><div class=bct>Each one is the <b>median over a 3-year window</b> &mdash; clouds average out into one clean image. Source: <b>7 Landsat bands (30&nbsp;m) + VIIRS night-lights</b>.</div></div>')
s9 += ('<div class=bcard><div class=bcl>WHY EIGHT, NOT THREE</div><div class=bct>Colour alone misses a lot. The CNN <b>fuses all eight</b> into one number &mdash; and night-lights will turn out to do most of the work, which matters later.</div></div>')
s9 += '</div>'
write_slide("s09_bands", "light", s9, s9_css)

# ---------- S10 · the model ----------
s10_css = """
.flow{position:absolute;left:118px;top:300px;width:1684px;display:flex;align-items:center;justify-content:center;gap:36px;z-index:5}
.inp{display:flex;flex-direction:column;align-items:center}
.stack{position:relative;width:150px;height:170px}
.stack i{position:absolute;width:118px;height:118px;border-radius:8px;border:1.5px solid #16b9d0;background:linear-gradient(135deg,#eafafc,#d6f0f4)}
.ilab{font-family:'JetBrains Mono';font-size:14px;color:#5f7782;margin-top:14px;text-align:center}
.arrow{font-size:40px;color:#16b9d0}
.box{background:#0f2636;color:#fff;border-radius:16px;padding:34px 40px;text-align:center;box-shadow:0 10px 26px rgba(15,38,54,.2)}
.box .t{font-family:'Bricolage Grotesque';font-weight:800;font-size:32px}
.box .s{font-size:16px;color:#9fc0cc;margin-top:6px}
.out{background:linear-gradient(120deg,#16b9d0,#1aa985);color:#fff;border-radius:16px;padding:34px 44px;text-align:center}
.out .t{font-family:'Bricolage Grotesque';font-weight:800;font-size:58px;line-height:1}
.out .s{font-size:16px;margin-top:6px;opacity:.95}
.blocks{position:absolute;left:118px;top:556px;width:1684px;display:flex;gap:28px;z-index:5}
.mb{flex:1;background:#fff;border:1px solid #dbe5ea;border-left:5px solid #16b9d0;border-radius:14px;padding:24px 28px;box-shadow:0 5px 16px rgba(20,40,55,.05)}
.mbl{font-family:'JetBrains Mono';font-size:14px;letter-spacing:2px;color:#1296ad;margin-bottom:10px}
.mbt{font-size:21px;line-height:1.46;color:#33454f}.mbt b{color:#0f2636}
.punch{position:absolute;left:118px;top:874px;width:1684px;text-align:center;font-family:'Bricolage Grotesque';font-weight:700;font-size:30px;color:#0f2636;z-index:5}
.punch em{font-style:normal;color:#0e8ba0}
"""
stack = '<div class=stack>' + ''.join(f'<i style="left:{i*4}px;top:{i*7}px"></i>' for i in range(7,-1,-1)) + '</div>'
s10 = header("The model, in <em>20 seconds</em>",
             "Deliberately ordinary &mdash; because the contribution of this work is not the network.",
             eyebrow="02 · THE INGREDIENTS", pageno="12 / 33")
s10 += ('<div class=flow>'
        f'<div class=inp>{stack}<div class=ilab>8 channels<br>224 &times; 224</div></div>'
        '<div class=arrow>&rarr;</div>'
        '<div class=box><div class=t>ResNet-18</div><div class=s>a standard CNN &middot; ~11M parameters</div></div>'
        '<div class=arrow>&rarr;</div>'
        '<div class=out><div class=t>0.83</div><div class=s>one number: predicted wealth</div></div>'
        '</div>')
s10 += '<div class=blocks>'
s10 += '<div class=mb><div class=mbl>WHAT IT IS</div><div class=mbt>An off-the-shelf <b>ResNet-18</b> &mdash; the same network used for ordinary image classification.</div></div>'
s10 += '<div class=mb><div class=mbl>WHAT WE CHANGED</div><div class=mbt>Its first layer accepts <b>8 channels, not 3</b>; its last layer outputs <b>one number</b> (regression), not 1,000 class labels.</div></div>'
s10 += '<div class=mb><div class=mbl>HOW WE TRAIN IT</div><div class=mbt style="font-size:16px;line-height:1.62"><div><b style="color:#16b9d0">&bull;</b> <b>MSE loss</b> &mdash; optimises r&sup2; directly</div><div><b style="color:#16b9d0">&bull;</b> <b>Adam, lr 3e-4</b> (cosine) &mdash; stable, low-tuning; cut from 1e-3</div><div><b style="color:#16b9d0">&bull;</b> <b>Batch 64</b> &mdash; fits the 8-channel tiles in GPU memory</div><div><b style="color:#16b9d0">&bull;</b> <b>Early-stop on val r&sup2;</b> &mdash; best-generalising model</div><div><b style="color:#16b9d0">&bull;</b> <b>From scratch</b> &mdash; pre-training doesn&rsquo;t help (backup)</div></div></div>'
s10 += '</div>'
s10 += '<div class=punch>The science is in the <em>data</em> and the <em>honest evaluation</em> &mdash; not the architecture.</div>'
write_slide("s10_model", "light", s10, s10_css)

for n in ["s09_bands","s10_model"]:
    render(n); print("rendered", n)
