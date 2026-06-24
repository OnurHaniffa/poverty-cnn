"""Act 3: S11 divider, S12 CV ladder, S13 metrics, S13B overfit check."""
import base64, pathlib
from lib import write_slide, render, header
from build_light_deck import GLOBE
DECK = pathlib.Path(__file__).parent
def blk(label, text):
    return f'<div class=blk><div class=bl>{label}</div><div class=bt>{text}</div></div>'
def img(p):
    return "data:image/png;base64," + base64.b64encode((DECK/p).read_bytes()).decode()

# ---------- S11 · divider ----------
s11_css = """
.gl{position:absolute;inset:0;z-index:0}
.signum{position:absolute;left:118px;top:300px;font-family:'Bricolage Grotesque';font-weight:800;font-size:300px;line-height:.8;color:transparent;background:linear-gradient(150deg,#12b3cc,#1aa07f 60%,#2a6cf0);-webkit-background-clip:text;background-clip:text;z-index:4}
.sigrule{position:absolute;left:134px;top:600px;width:120px;height:6px;border-radius:3px;background:linear-gradient(90deg,#16b9d0,#1aa985);z-index:5}
.sigttl{position:absolute;left:130px;top:640px;font-family:'Bricolage Grotesque';font-weight:800;font-size:74px;color:#0f2636;z-index:5}
.sigsub{position:absolute;left:134px;top:762px;font-size:30px;color:#516470;z-index:5;max-width:1000px;font-weight:500}
.sigsub em{font-style:normal;color:#0e8ba0;font-weight:700}
"""
s11 = f'<div class=gl>{GLOBE}</div><div class=pg>13 / 33</div><div class=eb>SECTION 03</div>'
s11 += '<div class=signum>03</div><div class=sigrule></div><div class=sigttl>Does it work?</div>'
s11 += '<div class=sigsub>First &mdash; <em>the honest test.</em> Because how you split the data decides whether the number is real.</div>'
write_slide("s11_doesitwork", "light", s11, s11_css)

# ---------- S12 · cross-validation ladder ----------
s12_css = """
.step{position:absolute;width:415px;border-radius:16px;padding:24px 28px;background:#fff;box-shadow:0 8px 22px rgba(20,40,55,.08);z-index:5}
.s1{left:96px;top:290px;border-top:5px solid #e0a02e}
.s2{left:770px;top:430px;border:2px solid #16b9d0;box-shadow:0 12px 28px rgba(22,185,208,.22)}
.s3{left:1210px;top:570px;border-top:5px solid #5b7384}
.reg{font-family:'JetBrains Mono';font-size:14px;letter-spacing:2px;color:#5f7782;margin-bottom:8px}
.r2{font-family:'Bricolage Grotesque';font-weight:800;font-size:60px;line-height:1}
.s1 .r2{color:#cf8a1e}.s2 .r2{color:transparent;background:linear-gradient(96deg,#12b3cc,#1aa07f);-webkit-background-clip:text;background-clip:text}.s3 .r2{color:#3c5163}
.r2 span{font-size:24px;color:#9fb1bd;font-family:'Hanken Grotesk';font-weight:600}
.mean{font-size:19px;color:#566b78;line-height:1.4;margin-top:8px}.mean b{color:#0f2636}
.badge{display:inline-block;margin-top:14px;font-family:'JetBrains Mono';font-size:13px;letter-spacing:1px;padding:6px 12px;border-radius:20px}
.bad{background:rgba(224,160,46,.14);color:#b8791a}
.good{background:rgba(22,185,208,.16);color:#0e8ba0}
.neu{background:#eef2f4;color:#5b7384}
.infl{position:absolute;left:536px;top:352px;width:208px;text-align:center;z-index:6}
.infl .a{font-family:'Bricolage Grotesque';font-weight:800;font-size:40px;color:#cf5b63}
.infl .l{font-size:15px;color:#8a5a2a;font-weight:600;line-height:1.3}
.axis{position:absolute;left:96px;top:828px;font-family:'JetBrains Mono';font-size:14px;letter-spacing:1px;color:#90a4af;z-index:5}
.note{position:absolute;left:118px;bottom:66px;width:1684px;font-size:23px;color:#33454f;z-index:5}.note b{color:#0e8ba0}
"""
s12 = header("Why a random split <em>lies</em> &mdash; and how we test honestly",
             "The more honest the test, the lower the score. We climb the ladder and report the honest rung, not the flattering one.",
             eyebrow="03 · DOES IT WORK?", pageno="14 / 33")
s12 += ('<div class="step s1"><div class=reg>RANDOM SPLIT</div><div class=r2>0.73<span> r&sup2;</span></div>'
        '<div class=mean>Shuffle all villages, test on held-out ones &mdash; the model can <b>memorise each country</b>. Flattering, but fake.</div>'
        '<div class="badge bad">&#10005; inflated</div></div>')
s12 += ('<div class="step s2"><div class=reg>LEAVE-COUNTRY-OUT</div><div class=r2>0.61<span> r&sup2;</span></div>'
        '<div class=mean>Hold out <b>whole countries</b> &mdash; test on places the model never saw. <b>Every number we report is this.</b></div>'
        '<div class="badge good">&#10003; what we report</div></div>')
s12 += ('<div class="step s3"><div class=reg>OUT-OF-DISTRIBUTION</div><div class=r2>0.48<span> r&sup2;</span></div>'
        '<div class=mean>Brand-new countries outside training entirely &mdash; ranking still transfers, r&sup2; drops further.</div>'
        '<div class="badge neu">hardest test</div></div>')
s12 += '<div class=infl><div class=a>&minus;20%</div><div class=l>the inflation we refuse to claim</div></div>'
s12 += '<div class=axis>EASIEST / MOST FLATTERING &nbsp;&mdash;&mdash;&mdash;&rarr;&nbsp; HARDEST / MOST HONEST</div>'
s12 += '<div class=note>Analogy: a random split is an exam with questions you <b>studied</b>; leave-country-out is questions you&rsquo;ve <b>never seen</b>. We grade ourselves on the second one.</div>'
write_slide("s12_cvladder", "light", s12, s12_css)

# ---------- S13B · overfit check ----------
ns=[4734,9469,14204,18939,23674]; rs=[0.453,0.500,0.511,0.541,0.567]
x0,x1,y0,y1=86,556,300,52; ymin,ymax=0.42,0.59
def X(n): return x0+(n-ns[0])/(ns[-1]-ns[0])*(x1-x0)
def Y(r): return y0-(r-ymin)/(ymax-ymin)*(y0-y1)
pts=[(X(n),Y(r)) for n,r in zip(ns,rs)]
poly=" ".join(f"{x:.0f},{y:.0f}" for x,y in pts)
area=f"{x0},{y0} "+poly+f" {x1},{y0}"
dots="".join(f'<circle cx="{x:.0f}" cy="{y:.0f}" r="6" fill="#16b9d0" stroke="#fff" stroke-width="2"/>' for x,y in pts)
yt="".join(f'<line x1="{x0}" y1="{Y(v):.0f}" x2="{x1}" y2="{Y(v):.0f}" stroke="#eef2f4"/><text x="{x0-12}" y="{Y(v)+4:.0f}" text-anchor="end" font-size="13" fill="#90a4af" font-family="JetBrains Mono">{v:.1f}</text>' for v in [0.45,0.50,0.55])
xt="".join(f'<text x="{X(n):.0f}" y="{y0+26}" text-anchor="middle" font-size="12" fill="#90a4af" font-family="JetBrains Mono">{n//1000}k</text>' for n in ns)
vlab="".join(f'<text x="{x:.0f}" y="{y-15:.0f}" text-anchor="middle" font-size="13" fill="#0e8ba0" font-weight="bold" font-family="Hanken Grotesk">{r:.2f}</text>' for (x,y),r in zip(pts,rs))
scaling=f'''<svg width="620" height="372" viewBox="0 0 620 372">{yt}
<polyline points="{area}" fill="#16b9d0" opacity="0.10"/><polyline points="{poly}" fill="none" stroke="#16b9d0" stroke-width="3"/>{dots}{vlab}{xt}
<text x="{(x0+x1)/2:.0f}" y="362" text-anchor="middle" font-size="14" fill="#566b78" font-family="Hanken Grotesk">training villages &rarr;</text>
<text x="22" y="176" text-anchor="middle" font-size="14" fill="#566b78" font-family="Hanken Grotesk" transform="rotate(-90 22 176)">validation r&sup2;</text></svg>'''

s13b_css = """
.lc{position:absolute;left:104px;top:286px;width:780px;background:#fff;border:1px solid #dbe5ea;border-radius:16px;padding:18px 22px 8px;box-shadow:0 5px 16px rgba(20,40,55,.05);z-index:5}
.lc img{height:360px;width:auto;max-width:100%;display:block;margin:0 auto}.lc .cap{font-family:'JetBrains Mono';font-size:13px;letter-spacing:1.5px;color:#5f7782;margin-bottom:6px}
.sc{position:absolute;right:104px;top:286px;width:700px;background:#fff;border:1px solid #dbe5ea;border-radius:16px;padding:18px 26px 12px;box-shadow:0 5px 16px rgba(20,40,55,.05);z-index:5}
.sc .cap{font-family:'JetBrains Mono';font-size:13px;letter-spacing:1.5px;color:#5f7782;margin-bottom:8px}
.sc svg{height:330px;width:auto;display:block;margin:2px auto 0}
.lblk{position:absolute;left:104px;top:720px;width:780px;z-index:5}
.rblk{position:absolute;right:104px;top:720px;width:700px;z-index:5}
.lblk .blk,.rblk .blk{padding-left:24px}.lblk .bt,.rblk .bt{font-size:21px;line-height:1.46}
.take{position:absolute;left:118px;bottom:60px;width:1684px;text-align:center;font-size:23px;color:#0e7c86;font-weight:600;z-index:5}.take b{color:#0f2636}
"""
s13b = header("Did it overfit? &mdash; <em>the honest check</em>",
              "Two questions: does training run away from validation, and would more data help? Both answered from the curves.",
              eyebrow="03 · DOES IT WORK?", pageno="17 / 33")
s13b += f'<div class=lc><div class=cap>LEARNING CURVE &mdash; TRAIN vs VALIDATION r&sup2; PER EPOCH</div><img src="{img("assets/curves/learning_curve.png")}"></div>'
s13b += f'<div class=sc><div class=cap>DATA-SCALING &mdash; VALIDATION r&sup2; vs TRAINING SIZE</div>{scaling}</div>'
s13b += '<div class=lblk>'+blk("WE STOP BEFORE IT DRIFTS","Training r&sup2; climbs to 0.81; validation plateaus near 0.55. We <b>early-stop at the best validation epoch</b> (dashed line) and only ever report validation / test &mdash; never the inflated training number.")+'</div>'
s13b += '<div class=rblk>'+blk("THE LIMIT IS DATA, NOT THE MODEL","Give it more villages and validation r&sup2; keeps rising &mdash; <b>0.45 &rarr; 0.57</b>. The model is <b>data-limited</b>, not memorising noise. (More data lifts the <i>average</i>, though &mdash; not the poorest, slide 26.) A fancier net wouldn&rsquo;t help either: ImageNet-pretrained scores <b>&minus;0.03 r&sup2;</b>.")+'</div>'
s13b += '<div class=take>&rarr; The train&ndash;validation gap is <b>controlled book-keeping</b> (early-stopping), and the ceiling is <b>set by data</b> &mdash; not overfitting.</div>'
write_slide("s13b_overfit", "light", s13b, s13b_css)

for n in ["s11_doesitwork","s12_cvladder","s13b_overfit"]:
    render(n); print("rendered", n)
