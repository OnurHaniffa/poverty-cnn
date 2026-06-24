"""Act 5 batch 1: S17 the turn, S18 cities>villages, S19 confidently wrong about the poorest."""
from lib import write_slide, render, header, method_note
from build_light_deck import GLOBE
CYAN="#16b9d0"; NAVY="#3c5163"; GREY="#c3ced4"
def blk(label, text):
    return f'<div class=blk><div class=bl>{label}</div><div class=bt>{text}</div></div>'

# ---------- S17 · the turn (divider) ----------
s17_css = """
.gl{position:absolute;inset:0;z-index:0}
.signum{position:absolute;left:118px;top:250px;font-family:'Bricolage Grotesque';font-weight:800;font-size:300px;line-height:.8;color:transparent;background:linear-gradient(150deg,#d98a3a,#cf5b63 70%);-webkit-background-clip:text;background-clip:text;z-index:4;opacity:.92}
.sigrule{position:absolute;left:134px;top:560px;width:120px;height:6px;border-radius:3px;background:linear-gradient(90deg,#16b9d0,#1aa985);z-index:5}
.sigttl{position:absolute;left:130px;top:600px;font-family:'Bricolage Grotesque';font-weight:800;font-size:72px;line-height:1.05;color:#0f2636;z-index:5;max-width:1180px}
.sigttl em{font-style:normal;color:#0e8ba0}
.sigsub{position:absolute;left:134px;top:828px;font-size:30px;color:#516470;z-index:5;max-width:1080px;font-weight:500}
"""
s17 = f'<div class=gl>{GLOBE}</div><div class=pg>21 / 33</div><div class=eb>SECTION 04 · THE TURN</div>'
s17 += '<div class=signum>04</div><div class=sigrule></div>'
s17 += '<div class=sigttl>But it fails the people<br>it&rsquo;s meant to <em>find.</em></div>'
s17 += '<div class=sigsub>It works &mdash; on average. Now we ask the question that actually matters: <b>who does it work for?</b></div>'
write_slide("s17_theturn", "light", s17, s17_css)

# ---------- S18 · cities > villages ----------
def vbars(panels, W=880, H=388):
    n=len(panels); pw=W/n; baseY=H-52; topY=64; bh=baseY-topY; bw=60; gap=30; parts=[]
    for pi,(title,bars,vmax,hl) in enumerate(panels):
        px=pi*pw
        if hl: parts.append(f'<rect x="{px+6:.0f}" y="44" width="{pw-12:.0f}" height="{H-58}" rx="12" fill="rgba(22,185,208,.07)"/>')
        parts.append(f'<text x="{px+pw/2:.0f}" y="30" text-anchor="middle" font-size="16" font-family="JetBrains Mono" fill="{"#0e8ba0" if hl else "#5f7782"}">{title}</text>')
        sx=px+pw/2-(bw*2+gap)/2
        for bi,(lab,val,color) in enumerate(bars):
            bx=sx+bi*(bw+gap); h=val/vmax*bh
            parts.append(f'<rect x="{bx:.0f}" y="{baseY-h:.0f}" width="{bw}" height="{h:.0f}" rx="6" fill="{color}"/>')
            parts.append(f'<text x="{bx+bw/2:.0f}" y="{baseY-h-12:.0f}" text-anchor="middle" font-size="18" font-weight="700" font-family="Hanken Grotesk" fill="#0f2636">{val:.2f}</text>')
            parts.append(f'<text x="{bx+bw/2:.0f}" y="{baseY+26:.0f}" text-anchor="middle" font-size="15" font-family="Hanken Grotesk" fill="#7a8d98">{lab}</text>')
    return f'<svg width="{W}" height="{H}" viewBox="0 0 {W} {H}">{"".join(parts)}</svg>'

chart18 = vbars([
    ("r&sup2;",[("urban",0.234,CYAN),("rural",0.125,NAVY)],0.30,False),
    ("MAE · lower better",[("urban",0.459,CYAN),("rural",0.441,NAVY)],0.55,False),
    ("SPEARMAN · ranking",[("urban",0.535,CYAN),("rural",0.384,NAVY)],0.65,True)])
s18_css = """
.chart{position:absolute;left:96px;top:300px;width:900px;background:#fff;border:1px solid #dbe5ea;border-radius:16px;padding:22px 24px 12px;box-shadow:0 6px 18px rgba(20,40,55,.06);z-index:5}
.chart .cap{font-family:'JetBrains Mono';font-size:14px;letter-spacing:1.5px;color:#5f7782;margin-bottom:6px}
.rcol{position:absolute;right:104px;top:308px;width:760px;z-index:5}
.rcol .blk{padding-left:24px}.rcol .bt{font-size:23px;line-height:1.5}
"""
s18 = header("It serves <em>cities better than villages</em>",
             "Same absolute error in both &mdash; but it can&rsquo;t <b>rank</b> rural villages, and most villages are rural.",
             eyebrow="04 · THE TURN", pageno="22 / 33")
s18 += f'<div class=chart><div class=cap>URBAN vs RURAL &mdash; ACROSS 23 COUNTRIES (n = 4,980 urban · 8,473 rural)</div>{chart18}</div>'
s18 += '<div class=rcol>'
s18 += blk("EQUAL ERROR, UNEQUAL RANKING","Absolute error is the same in town and country (MAE 0.46 vs 0.44). But <b>ranking collapses for rural villages</b> &mdash; Spearman <b>0.38</b> vs <b>0.54</b>.")
s18 += blk("WHY IT HAPPENS","Rural villages are bunched near the bottom (little spread to rank), <b>and</b> the signal the model leans on &mdash; night-lights &mdash; is mostly absent there.")
s18 += blk("WHO IT HURTS","The <b>rural poor</b> &mdash; the 8,473, the majority, the people targeting is meant to reach &mdash; are exactly who it ranks worst.")
s18 += '</div>'
s18 += method_note("For every held-out prediction we rejoined its DHS <b>urban/rural</b> tag, then computed each metric within each group &mdash; pooled across seeds, with bootstrap confidence intervals.")
write_slide("s18_cities_villages", "light", s18, s18_css)

# ---------- S19 · confidently wrong about the poorest ----------
biases=[0.62,0.43,0.26,0.11,0.00,-0.13,-0.24,-0.30,-0.35,-0.50]
def biaschart(b, W=900, H=430):
    x0=92; x1=W-40; n=len(b); step=(x1-x0)/n; bw=step*0.6; midY=190; vmax=0.7; H2=150
    parts=[f'<line x1="{x0}" y1="{midY}" x2="{x1}" y2="{midY}" stroke="#9fb1bd" stroke-width="1.5"/>']
    parts.append(f'<text x="{x0-12}" y="{midY-H2-6:.0f}" text-anchor="end" font-size="13" font-family="JetBrains Mono" fill="#16b9d0">+0.7</text>')
    parts.append(f'<text x="{x0-12}" y="{midY+5}" text-anchor="end" font-size="13" font-family="JetBrains Mono" fill="#7a8d98">0</text>')
    for i,v in enumerate(b):
        cx=x0+step*(i+0.5); h=abs(v)/vmax*H2
        if v>=0: y=midY-h; col=CYAN
        else: y=midY; col=NAVY
        parts.append(f'<rect x="{cx-bw/2:.0f}" y="{y:.0f}" width="{bw:.0f}" height="{max(h,1):.0f}" rx="4" fill="{col}"/>')
        vy=(midY-h-9) if v>=0 else (midY+h+20)
        parts.append(f'<text x="{cx:.0f}" y="{vy:.0f}" text-anchor="middle" font-size="14" font-weight="700" font-family="Hanken Grotesk" fill="{col}">{v:+.2f}</text>')
    parts.append(f'<text x="{x0+step*0.5:.0f}" y="{H-14}" text-anchor="middle" font-size="15" font-family="JetBrains Mono" fill="#7a8d98">POOREST</text>')
    parts.append(f'<text x="{x1-step*0.5:.0f}" y="{H-14}" text-anchor="middle" font-size="15" font-family="JetBrains Mono" fill="#7a8d98">RICHEST</text>')
    parts.append(f'<text x="{(x0+x1)/2:.0f}" y="{H-14}" text-anchor="middle" font-size="14" font-family="Hanken Grotesk" fill="#90a4af">true wealth decile &rarr;</text>')
    parts.append(f'<text x="{x0+step*1.4:.0f}" y="{midY-H2+4:.0f}" font-size="15" font-family="Hanken Grotesk" font-weight="600" fill="#0e8ba0">predicts the poor as RICHER than they are</text>')
    return f'<svg width="{W}" height="{H}" viewBox="0 0 {W} {H}">{"".join(parts)}</svg>'

chart19=biaschart(biases)
s19_css = """
.chart{position:absolute;left:96px;top:300px;width:920px;background:#fff;border:1px solid #dbe5ea;border-radius:16px;padding:22px 24px 10px;box-shadow:0 6px 18px rgba(20,40,55,.06);z-index:5}
.chart .cap{font-family:'JetBrains Mono';font-size:14px;letter-spacing:1.5px;color:#5f7782;margin-bottom:6px}
.slope{position:absolute;right:104px;top:300px;width:760px;background:#0f2636;border-radius:16px;padding:24px 30px;z-index:5;display:flex;align-items:center;gap:24px}
.slope .n{font-family:'Bricolage Grotesque';font-weight:800;font-size:64px;color:#3fe3f0;line-height:1}
.slope .t{font-size:19px;color:#cfe0e8;line-height:1.4}.slope .t b{color:#fff}
.rcol{position:absolute;right:104px;top:452px;width:760px;z-index:5}
.rcol .blk{padding-left:24px}.rcol .bt{font-size:22px;line-height:1.48}
"""
s19 = header("Confidently wrong about <em>the poorest</em>",
             "It doesn&rsquo;t just lose accuracy at the bottom &mdash; it is <b>systematically, predictably</b> wrong there.",
             eyebrow="04 · THE TURN", pageno="23 / 33")
s19 += f'<div class=chart><div class=cap>PREDICTION BIAS (PRED &minus; TRUE) BY TRUE-WEALTH DECILE</div>{chart19}</div>'
s19 += '<div class=slope><div class=n>0.60</div><div class=t>slope of predicted vs. true wealth.<br><b>1.0 would be perfect</b> &mdash; 0.60 means every prediction is pulled toward the middle.</div></div>'
s19 += '<div class=rcol>'
s19 += blk("REGRESSION TO THE MEAN","Because the model hedges toward the average, it <b>over-predicts the poorest by +0.62</b> and under-predicts the richest by &minus;0.50 &mdash; a clean, monotonic bias.")
s19 += blk("WHY IT&rsquo;S DANGEROUS","Used to find the neediest, it lifts the poorest <b>out of the danger zone</b> on paper &mdash; so the villages most needing aid look better off than they are.")
s19 += '</div>'
s19 += method_note("We regressed predicted on true wealth across all 13,453 held-out villages (the slope), and averaged the <b>signed</b> error within each true-wealth decile (the bars).")
write_slide("s19_poorest", "light", s19, s19_css)

for n in ["s17_theturn","s18_cities_villages","s19_poorest"]:
    render(n); print("rendered", n)
