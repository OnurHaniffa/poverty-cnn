"""S18, S19, S20, S22 rebuilt as cohesive stories: why -> did -> found -> means + verdict."""
from lib import write_slide, render, header, method_note, verdict
CYAN="#16b9d0"; NAVY="#3c5163"; GREY="#c3ced4"; CORAL="#d9747c"
def blk(label, text):
    return f'<div class=blk><div class=bl>{label}</div><div class=bt>{text}</div></div>'
RCOL = ".rcol{position:absolute;right:104px;top:300px;width:770px;height:442px;display:flex;flex-direction:column;justify-content:space-between;z-index:5}.rcol .blk{margin-bottom:0;padding-left:24px}.rcol .bt{font-size:20px;line-height:1.4}"
CHART = ".chart{position:absolute;left:96px;top:300px;width:892px;background:#fff;border:1px solid #dbe5ea;border-radius:16px;padding:22px 24px 12px;box-shadow:0 6px 18px rgba(20,40,55,.06);z-index:5}.chart .cap{font-family:'JetBrains Mono';font-size:14px;letter-spacing:1.5px;color:#5f7782;margin-bottom:6px}"

# ---------- S18 ----------
def vbars(panels, W=860, H=400):
    n=len(panels); pw=W/n; baseY=H-54; topY=64; bh=baseY-topY; bw=58; gap=28; parts=[]
    for pi,(title,bars,vmax,hl) in enumerate(panels):
        px=pi*pw
        if hl: parts.append(f'<rect x="{px+6:.0f}" y="42" width="{pw-12:.0f}" height="{H-56}" rx="12" fill="rgba(22,185,208,.08)"/>')
        parts.append(f'<text x="{px+pw/2:.0f}" y="30" text-anchor="middle" font-size="16" font-family="JetBrains Mono" fill="{"#0e8ba0" if hl else "#5f7782"}">{title}</text>')
        sx=px+pw/2-(bw*2+gap)/2
        for bi,(lab,val,color) in enumerate(bars):
            bx=sx+bi*(bw+gap); h=val/vmax*bh
            parts.append(f'<rect x="{bx:.0f}" y="{baseY-h:.0f}" width="{bw}" height="{h:.0f}" rx="6" fill="{color}"/>')
            parts.append(f'<text x="{bx+bw/2:.0f}" y="{baseY-h-11:.0f}" text-anchor="middle" font-size="18" font-weight="700" font-family="Hanken Grotesk" fill="#0f2636">{val:.2f}</text>')
            parts.append(f'<text x="{bx+bw/2:.0f}" y="{baseY+26:.0f}" text-anchor="middle" font-size="15" font-family="Hanken Grotesk" fill="#7a8d98">{lab}</text>')
    parts.append(f'<text x="{2.5*pw:.0f}" y="{H-6}" text-anchor="middle" font-size="14" font-style="italic" font-family="Hanken Grotesk" fill="#0e8ba0">the ranking gap is the whole story</text>')
    return f'<svg width="{W}" height="{H}" viewBox="0 0 {W} {H}">{"".join(parts)}</svg>'
c18=vbars([("r&sup2;",[("urban",0.234,CYAN),("rural",0.125,NAVY)],0.30,False),
           ("MAE · lower better",[("urban",0.459,CYAN),("rural",0.441,NAVY)],0.55,False),
           ("SPEARMAN · ranking",[("urban",0.535,CYAN),("rural",0.384,NAVY)],0.65,True)])
s=header("It serves <em>cities better than villages</em>",
         "It works on average &mdash; but averages hide who it fails. So we split every prediction into urban vs rural.",
         eyebrow="04 · THE TURN", pageno="22 / 33")
s+=f'<div class=chart><div class=cap>URBAN vs RURAL &mdash; ACROSS 23 COUNTRIES (n = 4,980 · 8,473)</div>{c18}</div><div class=rcol>'
s+=blk("WHAT WE DID","Split all 13,453 held-out predictions by their DHS <b>urban/rural</b> tag and scored each metric within each group.")
s+=blk("WHAT WE FOUND","Same <i>absolute</i> error (MAE 0.46 vs 0.44), but it can&rsquo;t <b>order</b> rural villages &mdash; Spearman <b>0.38 vs 0.54</b>, r&sup2; <b>0.12 vs 0.23</b>.")
s+=blk("WHAT IT MEANS","Rural villages bunch near the bottom (slide 10), so errors stay small but the model can&rsquo;t resolve <b>order within</b> that band &mdash; which is why targeting later misses ~half (slide 29).")
s+='</div>'
s+=verdict("Equal error, unequal ranking &mdash; it serves the <b>rural poor</b> worst.")
s+=method_note("Predictions tagged urban/rural by DHS cluster type; each metric computed within each group, pooled across seeds with bootstrap CIs.")
write_slide("s18_cities_villages","light",s,CHART+RCOL)

# ---------- S19 ----------
biases=[0.62,0.43,0.26,0.11,0.00,-0.13,-0.24,-0.30,-0.35,-0.50]
def biaschart(b,W=890,H=400):
    x0=96;x1=W-36;n=len(b);step=(x1-x0)/n;bw=step*0.6;midY=176;vmax=0.7;H2=130
    parts=[f'<line x1="{x0}" y1="{midY}" x2="{x1}" y2="{midY}" stroke="#9fb1bd" stroke-width="1.5"/>',
        f'<text x="{x0-6}" y="{midY-H2:.0f}" text-anchor="end" font-size="13" font-family="JetBrains Mono" fill="#16b9d0">+0.7</text>',
        f'<text x="{x0-6}" y="{midY+5}" text-anchor="end" font-size="13" font-family="JetBrains Mono" fill="#7a8d98">0</text>',
        f'<text x="{x0+step*1.7:.0f}" y="{midY-H2-8:.0f}" font-size="15" font-weight="700" font-family="Hanken Grotesk" fill="#0e8ba0">&uarr; predicted RICHER than reality</text>',
        f'<text x="{x1-6}" y="{midY+H2+24:.0f}" text-anchor="end" font-size="15" font-weight="700" font-family="Hanken Grotesk" fill="#3c5163">&darr; predicted poorer</text>']
    for i,v in enumerate(b):
        cx=x0+step*(i+0.5);h=abs(v)/vmax*H2
        y=(midY-h) if v>=0 else midY; col=CYAN if v>=0 else NAVY
        parts.append(f'<rect x="{cx-bw/2:.0f}" y="{y:.0f}" width="{bw:.0f}" height="{max(h,1):.0f}" rx="4" fill="{col}"/>')
        vy=(midY-h-9) if v>=0 else (midY+h+19)
        parts.append(f'<text x="{cx:.0f}" y="{vy:.0f}" text-anchor="middle" font-size="13.5" font-weight="700" font-family="Hanken Grotesk" fill="{col}">{v:+.2f}</text>')
    parts.append(f'<text x="{x0+step*0.5:.0f}" y="{H-12}" text-anchor="middle" font-size="14" font-family="JetBrains Mono" fill="#7a8d98">POOREST</text>')
    parts.append(f'<text x="{x1-step*0.5:.0f}" y="{H-12}" text-anchor="middle" font-size="14" font-family="JetBrains Mono" fill="#7a8d98">RICHEST</text>')
    parts.append(f'<text x="{(x0+x1)/2:.0f}" y="{H-12}" text-anchor="middle" font-size="13" font-family="Hanken Grotesk" fill="#90a4af">true wealth decile &rarr;</text>')
    return f'<svg width="{W}" height="{H}" viewBox="0 0 {W} {H}">{"".join(parts)}</svg>'
s=header("Confidently wrong about <em>the poorest</em>",
         "Does it fail the poor at random &mdash; or systematically? We measured its error at every wealth level.",
         eyebrow="04 · THE TURN", pageno="23 / 33")
s+=f'<div class=chart><div class=cap>PREDICTION BIAS (PRED &minus; TRUE) BY TRUE-WEALTH DECILE</div>{biaschart(biases)}</div><div class=rcol>'
s+=blk("WHAT WE DID","Sorted all villages poorest&rarr;richest into 10 bins and averaged the <b>signed</b> error (predicted &minus; true) in each.")
s+=blk("WHAT WE FOUND","A clean staircase: it over-predicts the poorest by <b>+0.62</b> and under-predicts the richest by <b>&minus;0.50</b> &mdash; slope 0.60.")
s+=blk("WHAT IT MEANS","Classic <b>regression to the mean</b>: it drags every guess toward the average, lifting the poorest <b>out of the danger zone</b> on paper.")
s+='</div>'
s+=verdict("It isn&rsquo;t noisy about the poor &mdash; it&rsquo;s <b>reliably wrong</b>, predicting them richer than they are.")
s+=method_note("Predicted regressed on true wealth across all 13,453 held-out villages (slope); signed error averaged within each true-wealth decile (bars).")
write_slide("s19_poorest","light",s,CHART+RCOL)


# ---------- S22 ----------
nl=[0.11,0.18,0.29,0.41,0.80,1.26,3.24,5.48,8.47,13.16]
def hockey(vals,W=890,H=400):
    x0=78;x1=W-36;y0=H-52;y1=44;vmax=14;n=len(vals)
    X=lambda i:x0+i/(n-1)*(x1-x0); Y=lambda v:y0-v/vmax*(y0-y1)
    pts=[(X(i),Y(v)) for i,v in enumerate(vals)]
    poly=" ".join(f"{x:.0f},{y:.0f}" for x,y in pts); area=f"{x0},{y0} "+poly+f" {x1},{y0}"
    parts=[f'<rect x="{x0}" y="{y1}" width="{X(4)-x0:.0f}" height="{y0-y1:.0f}" fill="rgba(60,81,99,.06)"/>',
        f'<polyline points="{area}" fill="#16b9d0" opacity="0.10"/>',f'<polyline points="{poly}" fill="none" stroke="#16b9d0" stroke-width="3.5"/>']
    for (x,y),v in zip(pts,vals): parts.append(f'<circle cx="{x:.0f}" cy="{y:.0f}" r="5" fill="#16b9d0" stroke="#fff" stroke-width="2"/>')
    parts.append(f'<text x="{pts[-1][0]-4:.0f}" y="{pts[-1][1]-12:.0f}" text-anchor="end" font-size="16" font-weight="700" font-family="Hanken Grotesk" fill="#0e8ba0">{nl[-1]:.1f}</text>')
    parts.append(f'<text x="{(x0+X(4))/2:.0f}" y="{y0-12:.0f}" text-anchor="middle" font-size="15" font-family="Hanken Grotesk" fill="#5b7384" font-weight="600">flat &amp; dark: the poorest half</text>')
    parts.append(f'<text x="{x0:.0f}" y="{H-12}" font-size="14" font-family="JetBrains Mono" fill="#7a8d98">POOREST</text>')
    parts.append(f'<text x="{x1:.0f}" y="{H-12}" text-anchor="end" font-size="14" font-family="JetBrains Mono" fill="#7a8d98">RICHEST</text>')
    parts.append(f'<text x="20" y="{(y0+y1)/2:.0f}" text-anchor="middle" font-size="14" font-family="Hanken Grotesk" fill="#566b78" transform="rotate(-90 20 {(y0+y1)/2:.0f})">night-light radiance</text>')
    return f'<svg width="{W}" height="{H}" viewBox="0 0 {W} {H}">{"".join(parts)}</svg>'
s=header("Why? Because <em>the lights are dark</em>",
         "Reweighting and uncertainty can&rsquo;t fix it &mdash; so <b>why</b> does it fail the poor at all? We looked at its strongest signal.",
         eyebrow="04 · THE TURN", pageno="27 / 33")
s+=f'<div class=chart><div class=cap>MEAN NIGHT-LIGHT RADIANCE BY TRUE-WEALTH DECILE</div>{hockey(nl)}</div><div class=rcol>'
s+=blk("WHAT WE DID","Averaged each village&rsquo;s <b>night-light brightness</b> by wealth decile, and measured the night-light&ndash;wealth correlation overall vs. within the poorest 30%.")
s+=blk("WHAT WE FOUND","Flat and near-zero across the poorest half, then it rockets up. Lights track wealth <b>0.76 overall</b> &mdash; but only <b>0.28</b> among the poorest.")
s+=blk("WHAT IT MEANS","The model&rsquo;s dominant signal is <b>physically absent</b> for the poor &mdash; it has nothing to read, so it hedges to the mean.")
s+='</div>'
s+=verdict("The main signal is <b>dark</b> for the poor &mdash; so the <b>ranking</b> failure is fundamental: no loss can fill a gap in the data.")
s+=method_note("Night-light band averaged over the 3-year composite; villages grouped into true-wealth deciles; rank correlation measured overall and within the poorest 30%.")
write_slide("s22_lights_dark","light",s,CHART+RCOL)

for n in ["s18_cities_villages","s19_poorest","s22_lights_dark"]:
    render(n); print("rendered",n)
