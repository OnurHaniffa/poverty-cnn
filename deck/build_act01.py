"""Act 0-1 slides: S2 what-is/isnt, S3 divider, S4 problem, S5 idea."""
from lib import write_slide, render, header, logic_blocks, stat_chips, globe

# ---------- S2 · What this is / isn't ----------
s2_css = """
.card{position:absolute;top:300px;width:780px;border-radius:18px;padding:38px 40px;background:rgba(10,26,40,.55);border:1px solid rgba(120,150,170,.18);z-index:5}
.cL{left:118px}.cR{left:1020px;background:rgba(40,20,30,.35)}
.ctag{font-family:'Bricolage Grotesque';font-weight:800;font-size:30px;letter-spacing:1px;margin-bottom:24px;display:flex;align-items:center;gap:14px}
.cIS .ctag{color:#3fe3f0}.cISNT .ctag{color:#ff9aa0}
.ic{width:38px;height:38px;border-radius:50%;display:flex;align-items:center;justify-content:center;font-size:23px;font-weight:700}
.cIS .ic{background:rgba(63,227,240,.16);color:#3fe3f0;border:1px solid rgba(63,227,240,.5)}
.cISNT .ic{background:rgba(255,140,150,.12);color:#ff9aa0;border:1px solid rgba(255,140,150,.45)}
.li{display:flex;gap:16px;margin-bottom:22px;align-items:flex-start}
.li .m{flex:none;width:26px;height:26px;border-radius:50%;display:flex;align-items:center;justify-content:center;font-size:15px;margin-top:2px}
.cIS .li .m{background:rgba(63,227,240,.16);color:#3fe3f0}.cISNT .li .m{background:rgba(255,140,150,.12);color:#ff9aa0}
.li .x{font-size:22px;line-height:1.4;color:#c9dae5}
.thesis{position:absolute;left:118px;bottom:66px;width:1684px;font-size:26px;color:#cfe8f0;z-index:5;font-weight:600}
.thesis b{background:linear-gradient(96deg,#7af0ff,#34d6c0);-webkit-background-clip:text;background-clip:text;color:transparent}
"""
def li(t): return f'<div class=li><div class=m>{"&#10003;" if "IS" else ""}</div><div class=x>{t}</div></div>'
s2 = header("What this is &mdash; and what it <em>isn&rsquo;t</em>",
            "We reproduce Yeh et&nbsp;al. (2020) faithfully &mdash; then run the audit the original paper didn&rsquo;t.",
            eyebrow="00 · FRAME", pageno="02 / 33")
s2 += '<div class="card cL cIS"><div class=ctag><span class=ic>&#10003;</span>IT IS</div>'
for t in ["A faithful <b>replication</b> of Yeh et&nbsp;al. (2020) &mdash; satellite imagery &rarr; village wealth.",
          "The <b>fairness, uncertainty &amp; generalization audit</b> the original never ran.",
          "Honest evaluation across <b>23 countries</b> &mdash; tested on countries the model never saw."]:
    s2 += f'<div class=li><div class=m>&#10003;</div><div class=x>{t}</div></div>'
s2 += '</div><div class="card cR cISNT"><div class=ctag><span class=ic>&#10005;</span>IT ISN&rsquo;T</div>'
for t in ["A new architecture or a <b>state-of-the-art</b> claim &mdash; the model is a plain ResNet-18.",
          "A <b>deploy-ready</b> poverty-targeting tool.",
          "A <b>criticism</b> of the original work &mdash; we build directly on it."]:
    s2 += f'<div class=li><div class=m>&#10005;</div><div class=x>{t}</div></div>'
s2 += '</div>'
s2 += '<div class=thesis>The point: <b>reproduce the headline &mdash; then stress-test it exactly where it matters most: the poorest.</b></div>'
write_slide("s02_frame", "dark", s2, s2_css)

# ---------- S3 · 01 The Promise (divider) ----------
s3_css = """
.signum{position:absolute;left:118px;top:300px;font-family:'Bricolage Grotesque';font-weight:800;font-size:300px;line-height:.8;color:transparent;background:linear-gradient(150deg,#1ff0ff,#1fae8b 60%,#2a6cf0);-webkit-background-clip:text;background-clip:text;z-index:4;opacity:.95}
.sigttl{position:absolute;left:130px;top:640px;font-family:'Bricolage Grotesque';font-weight:800;font-size:74px;color:#fff;z-index:5}
.sigrule{position:absolute;left:134px;top:600px;width:120px;height:6px;border-radius:3px;background:linear-gradient(90deg,#18cfe6,#34d6c0);box-shadow:0 0 22px rgba(24,207,230,.6);z-index:5}
.sigsub{position:absolute;left:134px;top:760px;font-size:30px;color:#a7c0d0;z-index:5;max-width:980px;font-weight:500}
.sigsub em{font-style:normal;color:#46e3f0}
"""
s3 = globe()
s3 += '<div class=pg>03 / 33</div><div class=eb>SECTION 01</div>'
s3 += '<div class=signum>01</div><div class=sigrule></div>'
s3 += '<div class=sigttl>The Promise</div>'
s3 += '<div class=sigsub>Can a <em>free satellite image</em> replace an expensive household survey?</div>'
write_slide("s03_promise", "dark", s3, s3_css)

# ---------- S4 · The Problem ----------
s4_css = """
.bigstat{position:absolute;right:118px;width:720px;z-index:5}
.bs1{top:300px}.bs2{top:560px}
.bsn{font-family:'Bricolage Grotesque';font-weight:800;font-size:120px;line-height:1;color:transparent;background:linear-gradient(96deg,#7af0ff,#34d6c0);-webkit-background-clip:text;background-clip:text}
.bsc{font-size:25px;color:#9fb4c6;margin-top:6px;max-width:660px}
.bsc b{color:#cfe8f0}
"""
s4 = header("Finding the poor is slow, costly, and <em>out of date</em>",
            "To help people in poverty, you first have to <b>find</b> them &mdash; and the ground-truth data to do that is scarce.",
            eyebrow="01 · THE PROMISE", pageno="04 / 33")
s4 += logic_blocks([
 ("WHY IT'S HARD", "Household surveys are the gold standard, but they&rsquo;re <b>expensive, slow, and sparse</b> &mdash; many regions are surveyed only once every several years, if at all."),
 ("THE OPPORTUNITY", "Satellites image <b>the entire planet, for free, repeatedly</b>. If a model can read wealth from that imagery, it&rsquo;s a scalable <b>complement</b> to surveys &mdash; filling the gaps in time and space."),
 ("THE STAKES", "These maps already steer <b>real aid</b>: Togo&rsquo;s Novissi program used satellite poverty maps to target emergency COVID cash (Aiken&nbsp;2022). When the map is wrong about the poorest, real people are missed."),
], 118, 300, 880)
s4 += ('<div class="bigstat bs1"><div class=bsn>1.8B</div>'
       '<div class=bsc>people live below the poverty line &mdash; and the data to <b>locate</b> them is scarce and stale</div></div>'
       '<div class="bigstat bs2"><div class=bsn>~5 yrs</div>'
       '<div class=bsc>typical gap between household surveys &mdash; <b>often far longer</b> in the poorest places</div></div>')
write_slide("s04_problem", "dark", s4, s4_css)

# ---------- S5 · The Idea (pipeline) ----------
s5_css = """
.pipe{position:absolute;left:118px;top:330px;width:1684px;display:flex;justify-content:space-between;z-index:5}
.step{width:300px;background:rgba(10,26,40,.5);border:1px solid rgba(70,227,240,.2);border-radius:16px;padding:24px 22px;position:relative}
.bdg{width:46px;height:46px;border-radius:12px;display:flex;align-items:center;justify-content:center;font-family:'Bricolage Grotesque';font-weight:800;font-size:24px;color:#04121c;background:linear-gradient(120deg,#19d8ef,#1fae8b);margin-bottom:16px}
.stt{font-family:'Bricolage Grotesque';font-weight:700;font-size:25px;color:#fff;margin-bottom:8px}
.sds{font-size:19px;color:#9fb4c6;line-height:1.4}
.conn{position:absolute;top:354px;height:2px;background:linear-gradient(90deg,rgba(70,227,240,.5),rgba(70,227,240,.15));z-index:4}
.trained{position:absolute;left:118px;top:560px;width:1684px;text-align:center;font-size:23px;color:#8fb6c8;z-index:5}
.trained b{color:#46e3f0}
"""
steps = [("1","GPS","Every surveyed village has a coordinate."),
         ("2","Satellite tile","Pull a 6.7&nbsp;km image around that point."),
         ("3","8 bands","Colour &middot; infrared &middot; thermal &middot; <b style='color:#cfe8f0'>night-lights</b>."),
         ("4","ResNet-18","A standard CNN reads all 8 channels together."),
         ("5","One number","It outputs the predicted <b style='color:#cfe8f0'>wealth</b>.")]
s5 = header("The idea: <em>one model, one number</em>",
            "The whole system in one line &mdash; then we spend the rest of the talk unpacking every piece honestly.",
            eyebrow="01 · THE PROMISE", pageno="05 / 33")
s5 += '<div class=pipe>'
for n,t,d in steps:
    s5 += f'<div class=step><div class=bdg>{n}</div><div class=stt>{t}</div><div class=sds>{d}</div></div>'
s5 += '</div>'
s5 += '<div class=trained>&#9650;&nbsp; all of it <b>trained to match the DHS wealth index</b> &mdash; the PCA &ldquo;answer key&rdquo; from the next section</div>'
s5 += logic_blocks([
 ("WHY IT CAN WORK", "Wealth has a <b>physical footprint</b> visible from above &mdash; dense metal roofs, paved roads, and electric light at night. The CNN learns to read those cues into a single score."),
], 118, 640, 1684)
write_slide("s05_idea", "dark", s5, s5_css)

for n in ["s02_frame","s03_promise","s04_problem","s05_idea"]:
    render(n); print("rendered", n)
