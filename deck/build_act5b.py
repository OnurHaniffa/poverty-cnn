"""Act 5 batch 2: S20 calibration!=equity, S21 nothing fixes it, S22 lights are dark."""
from lib import write_slide, render, header, method_note
CYAN="#16b9d0"; NAVY="#3c5163"; GREY="#c3ced4"; CORAL="#d9747c"
def blk(label, text):
    return f'<div class=blk><div class=bl>{label}</div><div class=bt>{text}</div></div>'

# ---------- S20 · calibration != equity ----------
def covchart(groups, W=900, H=400):
    # groups: [(method, cov_poor, cov_rich)]
    n=len(groups); pw=W/n; baseY=H-54; topY=70; bh=baseY-topY; bw=52; gap=26
    parts=[]
    ry=baseY-0.9*bh  # target line at 0.9
    parts.append(f'<line x1="40" y1="{ry:.0f}" x2="{W-20}" y2="{ry:.0f}" stroke="#1aa985" stroke-width="1.5" stroke-dasharray="6 6"/>')
    parts.append(f'<text x="44" y="{ry-8:.0f}" text-anchor="start" font-size="13" font-family="JetBrains Mono" fill="#1aa985">target 90% coverage</text>')
    for gi,(meth,cp,cr) in enumerate(groups):
        px=gi*pw; sx=px+pw/2-(bw*2+gap)/2
        for bi,(lab,val,color) in enumerate([("poorest 20%",cp,CYAN),("richest 20%",cr,NAVY)]):
            bx=sx+bi*(bw+gap); h=val*bh
            parts.append(f'<rect x="{bx:.0f}" y="{baseY-h:.0f}" width="{bw}" height="{h:.0f}" rx="6" fill="{color}"/>')
            parts.append(f'<text x="{bx+bw/2:.0f}" y="{baseY-h-11:.0f}" text-anchor="middle" font-size="17" font-weight="700" font-family="Hanken Grotesk" fill="#0f2636">{int(val*100)}%</text>')
            parts.append(f'<text x="{bx+bw/2:.0f}" y="{baseY+22:.0f}" text-anchor="middle" font-size="13" font-family="Hanken Grotesk" fill="#7a8d98">{lab}</text>')
        parts.append(f'<text x="{px+pw/2:.0f}" y="{H-12}" text-anchor="middle" font-size="16" font-family="JetBrains Mono" fill="#0f2636" font-weight="600">{meth}</text>')
    return f'<svg width="{W}" height="{H}" viewBox="0 0 {W} {H}">{"".join(parts)}</svg>'

chart20=covchart([("Deep ensemble",0.117,0.387),("MC-dropout",0.091,0.295),("Heteroscedastic",0.844,0.826)])
s20_css = """
.chart{position:absolute;left:96px;top:300px;width:900px;background:#fff;border:1px solid #dbe5ea;border-radius:16px;padding:22px 24px 10px;box-shadow:0 6px 18px rgba(20,40,55,.06);z-index:5}
.chart .cap{font-family:'JetBrains Mono';font-size:14px;letter-spacing:1.5px;color:#5f7782;margin-bottom:6px}
.rcol{position:absolute;right:104px;top:306px;width:760px;z-index:5}
.rcol .blk{padding-left:24px}.rcol .bt{font-size:21px;line-height:1.46}
"""
s20 = header("Calibration is <em>not equity</em>",
             "We asked three uncertainty methods: <b>do you know you&rsquo;re wrong about the poor?</b> None of them do.",
             eyebrow="04 · THE TURN", pageno="24 / 33")
s20 += f'<div class=chart><div class=cap>90% PREDICTION-INTERVAL COVERAGE &mdash; POOREST vs RICHEST 20%</div>{chart20}</div>'
s20 += '<div class=rcol>'
s20 += blk("OVERCONFIDENT ABOUT THE POOR","Ensemble &amp; MC-dropout cover only <b>12% / 9%</b> of the poorest (vs 39% / 30% of the rich) when they should cover 90% &mdash; tightest exactly where the model is most wrong.")
s20 += blk("EVEN THE CALIBRATED ONE IS BLIND","The heteroscedastic method equalises coverage (84% vs 83%) &mdash; but it hands the poor and the rich the <b>same confidence</b>, and its AURG &asymp; 0: uncertainty never ranks which predictions are wrong.")
s20 += blk("WHY: BIAS IS INVISIBLE","Error = aleatoric + <b>bias</b> + variance. Uncertainty estimates <b>variance</b>. The poorest&rsquo;s error is <b>bias</b> &mdash; structurally invisible to every one of these methods.")
s20 += '</div>'
s20 += method_note("Deep ensemble, MC-dropout and a heteroscedastic head, each evaluated for AURG (does uncertainty rank errors?) and prediction-interval coverage split by wealth quintile.")
write_slide("s20_calibration", "light", s20, s20_css)

# ---------- S21 · nothing fixes it ----------
def fixchart(bars, W=900, H=400):
    # bars: [(label, bias, r2)]
    n=len(bars); pw=W/n; baseY=H-78; topY=56; bh=baseY-topY; bw=92; vmax=1.5
    ly=baseY-0.618/vmax*bh
    parts=[f'<line x1="40" y1="{ly:.0f}" x2="{W-20}" y2="{ly:.0f}" stroke="#9fb1bd" stroke-width="1.4" stroke-dasharray="6 6"/>']
    for i,(lab,bias,r2) in enumerate(bars):
        cx=pw*(i+0.5); h=bias/vmax*bh; col=CORAL if bias>0.8 else (NAVY if i==0 else CYAN)
        parts.append(f'<rect x="{cx-bw/2:.0f}" y="{baseY-h:.0f}" width="{bw}" height="{h:.0f}" rx="6" fill="{col}"/>')
        parts.append(f'<text x="{cx:.0f}" y="{baseY-h-11:.0f}" text-anchor="middle" font-size="20" font-weight="700" font-family="Hanken Grotesk" fill="#0f2636">+{bias:.2f}</text>')
        parts.append(f'<text x="{cx:.0f}" y="{baseY+24:.0f}" text-anchor="middle" font-size="15" font-family="Hanken Grotesk" fill="#48606e" font-weight="600">{lab}</text>')
        parts.append(f'<text x="{cx:.0f}" y="{baseY+46:.0f}" text-anchor="middle" font-size="14" font-family="JetBrains Mono" fill="#90a4af">r&sup2; {r2:.2f}</text>')
    parts.append(f'<text x="20" y="{topY-30}" font-size="14" font-family="Hanken Grotesk" fill="#7a8d98">poorest-decile bias (want &darr; 0)</text>')
    return f'<svg width="{W}" height="{H}" viewBox="0 0 {W} {H}">{"".join(parts)}</svg>'

chart21=fixchart([("Standard model",0.618,0.569),("Loss reweighting",0.599,0.535),("Balanced-MSE",1.411,0.114)])
s21_css = """
.chart{position:absolute;left:96px;top:300px;width:900px;background:#fff;border:1px solid #dbe5ea;border-radius:16px;padding:24px 24px 12px;box-shadow:0 6px 18px rgba(20,40,55,.06);z-index:5}
.chart .cap{font-family:'JetBrains Mono';font-size:14px;letter-spacing:1.5px;color:#5f7782;margin-bottom:6px}
.rcol{position:absolute;right:104px;top:306px;width:760px;z-index:5}
.rcol .blk{padding-left:24px}.rcol .bt{font-size:21px;line-height:1.46}
"""
s21 = header("We tried to fix it. <em>Nothing works.</em>",
             "If the bias were a tuning problem, the standard fixes would move it. They don&rsquo;t.",
             eyebrow="04 · THE TURN", pageno="26 / 33")
s21 += f'<div class=chart><div class=cap>POOREST-DECILE BIAS &mdash; STANDARD MODEL vs EACH ATTEMPTED FIX</div>{chart21}</div>'
s21 += '<div class=rcol>'
s21 += blk("THE BASELINE = OUR STANDARD MODEL","The plain ResNet-18 (ordinary <b>MSE loss</b>, leave-country-out) behind every earlier number. It over-predicts the poorest decile by <b>+0.62</b> &mdash; the same bias as slide 22. Every fix is measured against this.")
s21 += blk("REWEIGHTING BARELY MOVES IT","Up-weighting poor villages (LDS) nudges it <b>+0.62 &rarr; +0.60</b> &mdash; while r&sup2; drops. A rounding error at a real accuracy cost.")
s21 += blk("BALANCED-MSE BACKFIRES","Forcing balance <b>destroys accuracy</b> (r&sup2; 0.57 &rarr; 0.11) <b>and</b> makes the bias worse (+1.41). The cure is worse than the disease.")
s21 += '</div>'
s21 += method_note("Baseline = the standard MSE-trained model. We retrained it with LDS reweighting (increasing strength) and a Balanced-MSE objective, then re-measured the <b>same</b> poorest-decile bias &mdash; and even tripling the training data barely moves it.")
write_slide("s21_nothing_works", "light", s21, s21_css)

# ---------- S22 · the lights are dark ----------
nl=[0.11,0.18,0.29,0.41,0.80,1.26,3.24,5.48,8.47,13.16]
def hockey(vals, W=920, H=420):
    x0=80; x1=W-40; y0=H-56; y1=46; vmax=14; n=len(vals)
    def X(i): return x0+i/(n-1)*(x1-x0)
    def Y(v): return y0-v/vmax*(y0-y1)
    pts=[(X(i),Y(v)) for i,v in enumerate(vals)]
    poly=" ".join(f"{x:.0f},{y:.0f}" for x,y in pts)
    area=f"{x0},{y0} "+poly+f" {x1},{y0}"
    parts=[f'<rect x="{x0}" y="{y1}" width="{X(4)-x0:.0f}" height="{y0-y1:.0f}" fill="rgba(60,81,99,.06)"/>']
    parts+=[f'<polyline points="{area}" fill="#16b9d0" opacity="0.10"/>',
            f'<polyline points="{poly}" fill="none" stroke="#16b9d0" stroke-width="3.5"/>']
    for (x,y),v in zip(pts,vals):
        parts.append(f'<circle cx="{x:.0f}" cy="{y:.0f}" r="5.5" fill="#16b9d0" stroke="#fff" stroke-width="2"/>')
    parts.append(f'<text x="{pts[0][0]+4:.0f}" y="{pts[0][1]-12:.0f}" font-size="14" font-family="Hanken Grotesk" font-weight="700" fill="#3c5163">{nl[0]:.1f}</text>')
    parts.append(f'<text x="{pts[-1][0]-4:.0f}" y="{pts[-1][1]-12:.0f}" text-anchor="end" font-size="16" font-family="Hanken Grotesk" font-weight="700" fill="#0e8ba0">{nl[-1]:.1f}</text>')
    parts.append(f'<text x="{(x0+X(4))/2:.0f}" y="{y0-14:.0f}" text-anchor="middle" font-size="15" font-family="Hanken Grotesk" fill="#5b7384" font-weight="600">flat &amp; dark: poorest half</text>')
    parts.append(f'<text x="{x0:.0f}" y="{H-14}" text-anchor="start" font-size="14" font-family="JetBrains Mono" fill="#7a8d98">POOREST</text>')
    parts.append(f'<text x="{x1:.0f}" y="{H-14}" text-anchor="end" font-size="14" font-family="JetBrains Mono" fill="#7a8d98">RICHEST</text>')
    parts.append(f'<text x="{(x0+x1)/2:.0f}" y="{H-14}" text-anchor="middle" font-size="14" font-family="Hanken Grotesk" fill="#90a4af">true wealth decile &rarr;</text>')
    parts.append(f'<text x="22" y="{(y0+y1)/2:.0f}" text-anchor="middle" font-size="14" font-family="Hanken Grotesk" fill="#566b78" transform="rotate(-90 22 {(y0+y1)/2:.0f})">night-light radiance</text>')
    return f'<svg width="{W}" height="{H}" viewBox="0 0 {W} {H}">{"".join(parts)}</svg>'

chart22=hockey(nl)
s22_css = """
.chart{position:absolute;left:96px;top:298px;width:940px;background:#fff;border:1px solid #dbe5ea;border-radius:16px;padding:22px 24px 10px;box-shadow:0 6px 18px rgba(20,40,55,.06);z-index:5}
.chart .cap{font-family:'JetBrains Mono';font-size:14px;letter-spacing:1.5px;color:#5f7782;margin-bottom:6px}
.corr{position:absolute;right:104px;top:298px;width:740px;display:flex;gap:18px;z-index:5}
.cc{flex:1;background:#0f2636;border-radius:14px;padding:20px 24px}
.cc .n{font-family:'Bricolage Grotesque';font-weight:800;font-size:46px;color:#3fe3f0;line-height:1}
.cc .c{font-size:16px;color:#a9bcc8;margin-top:6px;line-height:1.35}
.rcol{position:absolute;right:104px;top:470px;width:740px;z-index:5}
.rcol .blk{padding-left:24px}.rcol .bt{font-size:22px;line-height:1.5}
"""
s22 = header("Why? Because <em>the lights are dark</em>",
             "The whole failure in one chart: the signal the model leans on simply isn&rsquo;t there for the poor.",
             eyebrow="04 · THE TURN", pageno="27 / 33")
s22 += f'<div class=chart><div class=cap>MEAN NIGHT-LIGHT RADIANCE BY TRUE-WEALTH DECILE</div>{chart22}</div>'
s22 += ('<div class=corr>'
        '<div class=cc><div class=n>0.76</div><div class=c>night-lights vs wealth, <b>overall</b></div></div>'
        '<div class=cc><div class=n>0.28</div><div class=c>&hellip;but <b>within the poorest 30%</b></div></div></div>')
s22 += '<div class=rcol>'
s22 += blk("THE SIGNAL VANISHES AT THE BOTTOM","Night-lights track wealth strongly overall (0.76) &mdash; but among the poorest 30% they&rsquo;re all near-zero, so the correlation collapses to <b>0.28</b>. Lights can&rsquo;t tell the poor apart.")
s22 += blk("SO THE FAILURE IS FUNDAMENTAL","With its dominant signal physically absent, the model has nothing to read at the bottom &mdash; so it hedges to the mean. <b>It&rsquo;s a signal limit, not a bug.</b>")
s22 += '</div>'
s22 += method_note("We averaged each village&rsquo;s night-light band over the same 3-year composite, grouped villages into true-wealth deciles, and measured the night-light&ndash;wealth rank correlation overall vs. within the poorest 30%.")
write_slide("s22_lights_dark", "light", s22, s22_css)

for n in ["s20_calibration","s21_nothing_works","s22_lights_dark"]:
    render(n); print("rendered", n)
