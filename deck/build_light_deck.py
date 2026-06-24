"""Rebuild S1 title, S3 divider, S4 problem, S5 idea — ALL in the locked LIGHT scheme.
No dark slides anywhere. Near-white bg, navy text, cyan/teal accents (the PCA look)."""
import math, random
from lib import write_slide, render, header, logic_blocks

# ---- light data-globe (navy/teal wireframe + cyan cluster dots) ----
def make_globe(cx=1440, cy=545, R=300):
    random.seed(7)
    par=""
    for off in [-0.84,-0.5,-0.18,0.16,0.5,0.84]:
        oy=off*R; rx=math.sqrt(max(R*R-oy*oy,0)); ry=rx*0.30
        par+=f'<ellipse cx="{cx}" cy="{cy+oy:.1f}" rx="{rx:.1f}" ry="{ry:.1f}"/>'
    mer=""
    for lon in [-72,-43,-15,15,43,72]:
        rx=abs(R*math.sin(math.radians(lon)))
        mer+=f'<ellipse cx="{cx}" cy="{cy}" rx="{rx:.1f}" ry="{R}"/>'
    dots=""
    for i in range(80):
        lat=math.radians(random.uniform(-62,62)); lon=math.radians(random.uniform(-86,86))
        z=math.cos(lat)*math.cos(lon)
        if z<0.13: continue
        x=cx+R*math.cos(lat)*math.sin(lon); y=cy-R*math.sin(lat)
        r=1.6+2.6*z; op=0.40+0.55*z
        col="#0e9fb5" if random.random()>0.4 else "#1aa985"
        dots+=f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r:.2f}" fill="{col}" opacity="{op:.2f}"/>'
    sx,sy=cx-455,cy-150
    sat=(f'<g transform="translate({sx},{sy}) rotate(-18)"><rect x="-9" y="-7" width="18" height="14" rx="2" fill="#0f2636"/>'
         f'<rect x="-30" y="-3" width="18" height="6" fill="#16b9d0"/><rect x="12" y="-3" width="18" height="6" fill="#16b9d0"/></g>')
    return f'''<svg width="1920" height="1080" viewBox="0 0 1920 1080" style="position:absolute;inset:0;z-index:0">
<defs><radialGradient id="atm" cx="50%" cy="50%" r="50%"><stop offset="55%" stop-color="#bfe9f0" stop-opacity="0"/><stop offset="84%" stop-color="#7fd4e2" stop-opacity="0.45"/><stop offset="100%" stop-color="#7fd4e2" stop-opacity="0"/></radialGradient>
<radialGradient id="core" cx="42%" cy="38%" r="70%"><stop offset="0%" stop-color="#f3fbfc"/><stop offset="100%" stop-color="#dcecef"/></radialGradient></defs>
<circle cx="{cx}" cy="{cy}" r="{R+30}" fill="url(#atm)"/>
<circle cx="{cx}" cy="{cy}" r="{R}" fill="url(#core)"/>
<g fill="none" stroke="#2c7589" stroke-width="1" opacity="0.5">{par}{mer}</g>
<circle cx="{cx}" cy="{cy}" r="{R}" fill="none" stroke="#15708a" stroke-width="1.4" opacity="0.65"/>
{dots}
<ellipse cx="{cx}" cy="{cy}" rx="468" ry="158" fill="none" stroke="#16b9d0" stroke-width="1.2" opacity="0.4" transform="rotate(-18 {cx} {cy})"/>
{sat}</svg>'''

GLOBE = make_globe()

# ---------- S1 · TITLE (light) ----------
s1_css = """
.gl{position:absolute;inset:0;z-index:0}
.content{position:absolute;left:130px;top:0;height:100%;width:1020px;display:flex;flex-direction:column;justify-content:center;z-index:5}
.eb{position:static;font-family:'JetBrains Mono';font-size:20px;letter-spacing:6px;color:#1296ad;margin-bottom:28px}
.t{font-family:'Bricolage Grotesque';font-weight:800;font-size:92px;line-height:1.02;color:#0f2636;letter-spacing:-1.5px}
.t .g{background:linear-gradient(96deg,#12b3cc,#1aa07f);-webkit-background-clip:text;background-clip:text;color:transparent}
.t .d{color:#16b9d0}
.rule{width:96px;height:6px;background:linear-gradient(90deg,#16b9d0,#1aa985);margin:40px 0 30px;border-radius:3px}
.sub{font-size:26px;line-height:1.5;color:#516470;max-width:720px}
.foot{position:absolute;left:130px;bottom:76px;font-size:22px;color:#16303f;font-weight:600;z-index:5}
.foot span{color:#7c8d99;font-weight:400}
.pgm{position:absolute;left:130px;top:62px;font-family:'JetBrains Mono';font-size:19px;letter-spacing:3px;color:#8aa0ad;z-index:5}
.chip{position:absolute;right:120px;bottom:70px;font-family:'JetBrains Mono';font-size:15px;letter-spacing:2px;color:#0e8ba0;border:1px solid rgba(18,150,173,.4);padding:8px 14px;border-radius:30px;background:rgba(255,255,255,.7);z-index:5}
"""
s1 = f'<div class=gl>{GLOBE}</div>'
s1 += '<div class=pgm>01 / 33</div>'
s1 += ('<div class=content><div class=eb>ACIBADEM MAA&Uuml; &nbsp;/&nbsp; ML&middot;DL INTERNSHIP &nbsp;/&nbsp; 2026</div>'
       '<div class=t>Predicting Village<br><span class=g>Wealth from Space</span><span class=d>.</span></div>'
       '<div class=rule></div>'
       '<div class=sub>A fairness &amp; uncertainty audit of satellite poverty mapping &mdash; across 23 sub-Saharan African countries.</div></div>')
s1 += '<div class=chip>23 COUNTRIES &middot; 36,090 VILLAGES</div>'
s1 += '<div class=foot>Onur Haniffa <span>&nbsp;&middot;&nbsp; Advisor: Dr. Seda Nilg&uuml;n Dumlu</span></div>'
write_slide("s01_title", "light", s1, s1_css)

# ---------- S3 · DIVIDER (light) ----------
s3_css = """
.gl{position:absolute;inset:0;z-index:0}
.signum{position:absolute;left:118px;top:300px;font-family:'Bricolage Grotesque';font-weight:800;font-size:300px;line-height:.8;color:transparent;background:linear-gradient(150deg,#12b3cc,#1aa07f 60%,#2a6cf0);-webkit-background-clip:text;background-clip:text;z-index:4}
.sigrule{position:absolute;left:134px;top:600px;width:120px;height:6px;border-radius:3px;background:linear-gradient(90deg,#16b9d0,#1aa985);z-index:5}
.sigttl{position:absolute;left:130px;top:640px;font-family:'Bricolage Grotesque';font-weight:800;font-size:74px;color:#0f2636;z-index:5}
.sigsub{position:absolute;left:134px;top:762px;font-size:30px;color:#516470;z-index:5;max-width:980px;font-weight:500}
.sigsub em{font-style:normal;color:#0e8ba0;font-weight:700}
"""
s3 = f'<div class=gl>{GLOBE}</div>'
s3 += '<div class=pg>03 / 33</div><div class=eb>SECTION 01</div>'
s3 += '<div class=signum>01</div><div class=sigrule></div><div class=sigttl>The Promise</div>'
s3 += '<div class=sigsub>Can a <em>free satellite image</em> replace an expensive household survey?</div>'
write_slide("s03_promise", "light", s3, s3_css)

# ---------- S4 · PROBLEM (light) ----------
s4_css = """
.bigstat{position:absolute;right:118px;width:720px;z-index:5}
.bs1{top:300px}.bs2{top:560px}
.bsn{font-family:'Bricolage Grotesque';font-weight:800;font-size:118px;line-height:1;color:transparent;background:linear-gradient(96deg,#12b3cc,#1aa07f);-webkit-background-clip:text;background-clip:text}
.bsc{font-size:25px;color:#566b78;margin-top:6px;max-width:660px}.bsc b{color:#0f2636}
"""
s4 = header("Finding the poor is slow, costly, and <em>out of date</em>",
            "To help people in poverty, you first have to <b>find</b> them &mdash; and the ground-truth data to do that is scarce.",
            eyebrow="01 · THE PROMISE", pageno="04 / 33")
s4 += logic_blocks([
 ("WHY IT'S HARD","Household surveys are the gold standard, but they&rsquo;re <b>expensive, slow, and sparse</b> &mdash; many regions are surveyed only once every several years, if at all."),
 ("THE OPPORTUNITY","Satellites image <b>the entire planet, for free, repeatedly</b>. If a model can read wealth from that imagery, it&rsquo;s a scalable <b>complement</b> to surveys &mdash; filling the gaps in time and space."),
 ("THE STAKES","These maps already steer <b>real aid</b>: Togo&rsquo;s Novissi program used satellite poverty maps to target emergency COVID cash (Aiken&nbsp;2022). When the map is wrong about the poorest, real people are missed."),
], 118, 300, 880)
s4 += ('<div class="bigstat bs1"><div class=bsn>1.8B</div><div class=bsc>people live below the poverty line &mdash; and the data to <b>locate</b> them is scarce and stale</div></div>'
       '<div class="bigstat bs2"><div class=bsn>~5 yrs</div><div class=bsc>typical gap between household surveys &mdash; <b>often far longer</b> in the poorest places</div></div>')
write_slide("s04_problem", "light", s4, s4_css)

# ---------- S5 · IDEA / pipeline (light) ----------
s5_css = """
.pipe{position:absolute;left:118px;top:330px;width:1684px;display:flex;justify-content:space-between;z-index:5}
.step{width:300px;background:#fff;border:1px solid #d8e3e8;border-top:4px solid #16b9d0;border-radius:16px;padding:24px 22px;box-shadow:0 5px 16px rgba(20,40,55,.05)}
.bdg{width:46px;height:46px;border-radius:12px;display:flex;align-items:center;justify-content:center;font-family:'Bricolage Grotesque';font-weight:800;font-size:24px;color:#fff;background:linear-gradient(120deg,#16b9d0,#1aa985);margin-bottom:16px}
.stt{font-family:'Bricolage Grotesque';font-weight:700;font-size:25px;color:#0f2636;margin-bottom:8px}
.sds{font-size:19px;color:#566b78;line-height:1.4}.sds b{color:#0f2636}
.trained{position:absolute;left:118px;top:560px;width:1684px;text-align:center;font-size:23px;color:#4a6472;z-index:5}.trained b{color:#0e8ba0}
"""
steps=[("1","GPS","Every surveyed village has a coordinate."),
       ("2","Satellite tile","Pull a 6.7&nbsp;km image around that point."),
       ("3","8 bands","Colour &middot; infrared &middot; thermal &middot; <b>night-lights</b>."),
       ("4","ResNet-18","A standard CNN reads all 8 channels together."),
       ("5","One number","It outputs the predicted <b>wealth</b>.")]
s5 = header("The idea: <em>one model, one number</em>",
            "The whole system in one line &mdash; then we spend the rest of the talk unpacking every piece honestly.",
            eyebrow="01 · THE PROMISE", pageno="05 / 33")
s5 += '<div class=pipe>'
for n,t,d in steps:
    s5 += f'<div class=step><div class=bdg>{n}</div><div class=stt>{t}</div><div class=sds>{d}</div></div>'
s5 += '</div><div class=trained>&#9650;&nbsp; all of it <b>trained to match the DHS wealth index</b> &mdash; the PCA &ldquo;answer key&rdquo; from the next section</div>'
s5 += logic_blocks([("WHY IT CAN WORK","Wealth has a <b>physical footprint</b> visible from above &mdash; dense metal roofs, paved roads, and electric light at night. The CNN learns to read those cues into a single score."),], 118, 640, 1684)
write_slide("s05_idea", "light", s5, s5_css)

for n in ["s01_title","s03_promise","s04_problem","s05_idea"]:
    render(n); print("rendered", n)
