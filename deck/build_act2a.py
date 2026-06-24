"""Act 2 start: S6 divider (The Ingredients) + S7 DHS answer-key (with GPS-fuzz diagram)."""
from lib import write_slide, render, header
from build_light_deck import GLOBE

def blk(label, text):
    return f'<div class=blk><div class=bl>{label}</div><div class=bt>{text}</div></div>'

# ---------- S6 · DIVIDER 02 ----------
s6_css = """
.gl{position:absolute;inset:0;z-index:0}
.signum{position:absolute;left:118px;top:300px;font-family:'Bricolage Grotesque';font-weight:800;font-size:300px;line-height:.8;color:transparent;background:linear-gradient(150deg,#12b3cc,#1aa07f 60%,#2a6cf0);-webkit-background-clip:text;background-clip:text;z-index:4}
.sigrule{position:absolute;left:134px;top:600px;width:120px;height:6px;border-radius:3px;background:linear-gradient(90deg,#16b9d0,#1aa985);z-index:5}
.sigttl{position:absolute;left:130px;top:640px;font-family:'Bricolage Grotesque';font-weight:800;font-size:74px;color:#0f2636;z-index:5}
.sigsub{position:absolute;left:134px;top:762px;font-size:30px;color:#516470;z-index:5;max-width:1000px;font-weight:500}
.sigsub em{font-style:normal;color:#0e8ba0;font-weight:700}
"""
s6 = f'<div class=gl>{GLOBE}</div><div class=pg>06 / 33</div><div class=eb>SECTION 02</div>'
s6 += '<div class=signum>02</div><div class=sigrule></div><div class=sigttl>The Ingredients</div>'
s6 += '<div class=sigsub>Where the wealth number actually comes from &mdash; <em>the data, the label, and the image.</em></div>'
write_slide("s06_ingredients", "light", s6, s6_css)

# ---------- S7 · DHS answer key (left logic, right fuzz diagram) ----------
cx, cy = 1410, 590; T = 540
fuzz_r = 150
diagram = f'''<svg width="780" height="700" viewBox="0 0 780 700" style="position:absolute;right:90px;top:262px;z-index:5">
<defs><pattern id="g" width="45" height="45" patternUnits="userSpaceOnUse"><path d="M45 0H0V45" fill="none" stroke="#d4e4e8" stroke-width="1"/></pattern></defs>
<rect x="120" y="70" width="{T}" height="{T}" rx="10" fill="#eef7f9" stroke="#16b9d0" stroke-width="2.5"/>
<rect x="120" y="70" width="{T}" height="{T}" rx="10" fill="url(#g)"/>
<line x1="120" y1="48" x2="660" y2="48" stroke="#1296ad" stroke-width="1.5"/>
<line x1="120" y1="42" x2="120" y2="54" stroke="#1296ad" stroke-width="1.5"/><line x1="660" y1="42" x2="660" y2="54" stroke="#1296ad" stroke-width="1.5"/>
<text x="390" y="36" text-anchor="middle" font-family="JetBrains Mono" font-size="17" fill="#0e8ba0" letter-spacing="2">6.7 km TILE</text>
<circle cx="390" cy="340" r="{fuzz_r}" fill="rgba(26,169,133,.08)" stroke="#1aa985" stroke-width="2" stroke-dasharray="7 7"/>
<text x="390" y="218" text-anchor="middle" font-family="Hanken Grotesk" font-size="16" fill="#1aa985">privacy displacement (up to 5 km)</text>
<line x1="390" y1="340" x2="498" y2="268" stroke="#0f2636" stroke-width="1.5" stroke-dasharray="4 4"/>
<circle cx="390" cy="340" r="9" fill="#0f2636"/>
<text x="372" y="372" text-anchor="end" font-family="Hanken Grotesk" font-weight="600" font-size="17" fill="#0f2636">true village</text>
<circle cx="498" cy="268" r="10" fill="#16b9d0" stroke="#fff" stroke-width="2"/>
<text x="516" y="266" font-family="Hanken Grotesk" font-weight="600" font-size="17" fill="#0e8ba0">recorded GPS</text>
<text x="516" y="288" font-family="Hanken Grotesk" font-size="15" fill="#566b78">(fuzzed for privacy)</text>
<text x="390" y="648" text-anchor="middle" font-family="Hanken Grotesk" font-size="18" fill="#3a4c56">A wide tile still contains the real village &mdash; despite the shift.</text>
</svg>'''

s7_css = """
.lcol{position:absolute;left:118px;top:288px;width:780px;height:700px;display:flex;flex-direction:column;justify-content:space-between;z-index:5}
.lcol .blk{margin-bottom:0;padding-left:24px}
.lcol .bl{font-size:16px;letter-spacing:2.5px;margin-bottom:9px}
.lcol .bt{font-size:25px;line-height:1.5}
"""
s7 = header("The answer key: <em>DHS household surveys</em>",
            "Our model is only as good as its ground truth &mdash; so where does the &ldquo;true&rdquo; wealth come from?",
            eyebrow="02 · THE INGREDIENTS", pageno="07 / 33")
s7 += '<div class=lcol>'
s7 += blk("WHAT IT IS", "The <b>Demographic &amp; Health Surveys</b> are gold-standard, nationally-representative household surveys. A survey <b>cluster &asymp; a village</b> (~25 households), and each cluster has a <b>GPS coordinate</b>.")
s7 += blk("THE DELIBERATE TWIST", "For privacy, DHS <b>randomly shifts</b> each GPS &mdash; up to <b>2 km</b> in cities, <b>5 km</b> in rural areas. So the coordinate we get is deliberately <b>imprecise</b>.")
s7 += blk("HOW WE HANDLE IT", "That&rsquo;s exactly why our tile is a wide <b>6.7 km box</b> &mdash; big enough that the true village is still inside it, even after the shift. We trade resolution for <b>guaranteed coverage</b>.")
s7 += '</div>'
s7 += diagram
write_slide("s07_dhs", "light", s7, s7_css)

for n in ["s06_ingredients","s07_dhs"]:
    render(n); print("rendered", n)
