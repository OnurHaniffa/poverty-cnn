"""Split uncertainty slide into two: 24a poses the puzzle, 24b delivers the bias-vs-variance twist."""
from lib import write_slide, render, header, verdict, method_note
CYAN="#16b9d0"; NAVY="#3c5163"

# ---------- 24a · does it know when it's wrong? ----------
def covchart(groups,W=900,H=400):
    n=len(groups);pw=W/n;baseY=H-54;topY=78;bh=baseY-topY;bw=52;gap=26;parts=[]
    ry=baseY-0.9*bh
    parts.append(f'<line x1="36" y1="{ry:.0f}" x2="{W-16}" y2="{ry:.0f}" stroke="#1aa985" stroke-width="1.5" stroke-dasharray="6 6"/>')
    parts.append(f'<text x="40" y="{ry-9:.0f}" font-size="13" font-family="JetBrains Mono" fill="#1aa985">a good 90% interval covers 90%</text>')
    for gi,(meth,cp,cr) in enumerate(groups):
        px=gi*pw;sx=px+pw/2-(bw*2+gap)/2
        for bi,(lab,val,color) in enumerate([("poorest 20%",cp,CYAN),("richest 20%",cr,NAVY)]):
            bx=sx+bi*(bw+gap);h=val*bh
            parts.append(f'<rect x="{bx:.0f}" y="{baseY-h:.0f}" width="{bw}" height="{h:.0f}" rx="6" fill="{color}"/>')
            parts.append(f'<text x="{bx+bw/2:.0f}" y="{baseY-h-10:.0f}" text-anchor="middle" font-size="17" font-weight="700" font-family="Hanken Grotesk" fill="#0f2636">{int(val*100)}%</text>')
            parts.append(f'<text x="{bx+bw/2:.0f}" y="{baseY+22:.0f}" text-anchor="middle" font-size="13" font-family="Hanken Grotesk" fill="#7a8d98">{lab}</text>')
        parts.append(f'<text x="{px+pw/2:.0f}" y="{H-10}" text-anchor="middle" font-size="15.5" font-family="JetBrains Mono" fill="#0f2636" font-weight="600">{meth}</text>')
    return f'<svg width="{W}" height="{H}" viewBox="0 0 {W} {H}">{"".join(parts)}</svg>'

css_a = """
.methods{position:absolute;left:96px;top:304px;width:660px;z-index:5}
.met{background:#fff;border:1px solid #dbe5ea;border-left:4px solid #16b9d0;border-radius:12px;padding:18px 24px;margin-bottom:18px;box-shadow:0 4px 14px rgba(20,40,55,.05)}
.met .mn{font-family:'JetBrains Mono';font-size:16px;letter-spacing:1.5px;color:#0e8ba0;margin-bottom:6px}
.met .mt{font-size:19px;color:#42555f;line-height:1.42}.met .mt b{color:#0f2636}
.chart{position:absolute;right:92px;top:312px;width:1000px;background:#fff;border:1px solid #dbe5ea;border-radius:16px;padding:20px 24px 8px;box-shadow:0 6px 18px rgba(20,40,55,.06);z-index:5}
.chart .cap{font-family:'JetBrains Mono';font-size:13.5px;letter-spacing:.8px;color:#5f7782;margin-bottom:4px;line-height:1.4}
"""
ca=covchart([("Deep ensemble",0.117,0.387),("MC-dropout",0.091,0.295),("Heteroscedastic",0.844,0.826)])
s=header("Does it even <em>know</em> when it&rsquo;s wrong?",
         "Last hope: if it must fail the poor, can it at least flag its own doubt? We gave it three ways to express uncertainty, then checked its <b>90% error bars</b>.",
         eyebrow="04 · THE TURN", pageno="24 / 33")
s+='<div class=methods>'
s+='<div class=met><div class=mn>DEEP ENSEMBLE</div><div class=mt>Train <b>5 copies</b> of the model; where they <b>disagree</b>, it&rsquo;s unsure.</div></div>'
s+='<div class=met><div class=mn>MC-DROPOUT</div><div class=mt>Run <b>one</b> model many times with random neurons switched off; the <b>spread</b> is its doubt.</div></div>'
s+='<div class=met><div class=mn>HETEROSCEDASTIC</div><div class=mt>The model predicts its <b>own error bar</b> for each village.</div></div>'
s+='</div>'
s+=f'<div class=chart><div class=cap>90% COVERAGE = DOES THE 90% ERROR BAR ACTUALLY CONTAIN THE TRUTH 90% OF THE TIME?</div>{ca}</div>'
s+=verdict("Ensemble &amp; MC-dropout cover just <b>9&ndash;12%</b> of the poorest &mdash; overconfident. Heteroscedastic covers <b>84% vs 83%</b> &mdash; calibrated, and equal. So &mdash; problem solved? &rarr;")
s+=method_note("Three uncertainty heads; 90% prediction-interval coverage measured separately for the poorest and richest 20%.")
write_slide("s20a_calibration","light",s,css_a)

# ---------- 24b · calibration is not equity ----------
css_b = """
.catch{position:absolute;left:118px;top:290px;width:1684px;background:#fff;border:1px solid #dbe5ea;border-left:5px solid #d9747c;border-radius:14px;padding:22px 30px;box-shadow:0 5px 16px rgba(20,40,55,.05);z-index:5}
.catch .h{font-family:'JetBrains Mono';font-size:15px;letter-spacing:1.5px;color:#c0606a;margin-bottom:8px}
.catch .t{font-size:21px;color:#42555f;line-height:1.48}.catch .t b{color:#0f2636}
.formula{position:absolute;left:118px;top:474px;width:1684px;text-align:center;z-index:5}
.formula .fb{display:inline-block;background:#0f2636;border-radius:16px;padding:20px 50px;font-family:'Bricolage Grotesque';font-weight:800;font-size:44px;color:#9fb6c2}
.formula .fb b{color:#3fe3f0}.formula .fb .bias{color:#ff8a6b}
.two{position:absolute;left:118px;top:598px;width:1684px;display:flex;gap:30px;z-index:5}
.two .bx{flex:1;background:#fff;border:1px solid #dbe5ea;border-radius:14px;padding:24px 30px;box-shadow:0 5px 16px rgba(20,40,55,.05)}
.two .bh{font-family:'JetBrains Mono';font-size:15px;letter-spacing:1.5px;color:#1296ad;margin-bottom:8px}
.two .bt{font-size:20px;color:#42555f;line-height:1.5}.two .bt b{color:#0f2636}
"""
s=header("Calibration is <em>not equity</em>",
         "The calibrated method looked like it fixed it. It didn&rsquo;t &mdash; and why is the most important result in this work.",
         eyebrow="04 · THE TURN", pageno="25 / 33")
s+='<div class=catch><div class=h>THE CATCH &mdash; AURG &asymp; 0</div><div class=t><b>AURG</b> (<i>Area Under the Risk-coverage Gain</i>) is how we test whether the model&rsquo;s uncertainty can <b>find its own mistakes</b> &mdash; do its &lsquo;most-unsure&rsquo; guesses beat random at flagging the wrong ones? For all three methods it&rsquo;s &asymp; 0. So even the &lsquo;calibrated&rsquo; one hands the poor and the rich the <b>same</b> confidence &mdash; it can never single out the failures.</div></div>'
s+='<div class=formula><div class=fb>error &nbsp;=&nbsp; noise &nbsp;+&nbsp; <span class=bias>bias</span> &nbsp;+&nbsp; <b>variance</b></div></div>'
s+='<div class=two>'
s+='<div class=bx><div class=bh>UNCERTAINTY ONLY SEES VARIANCE</div><div class=bt>All three methods measure <b>variance</b> &mdash; how much predictions <b>wiggle</b> when you perturb the model. But for the poor, every model <b>agrees</b> on the same over-prediction, so variance is low and it looks <b>confident</b>.</div></div>'
s+='<div class=bx><div class=bh>THE POOR&rsquo;S ERROR IS BIAS</div><div class=bt>A <b>systematic, confident offset</b> &mdash; the staircase &mdash; not wiggle. Bias is <b>structurally invisible</b> to every uncertainty method, however well-calibrated.</div></div>'
s+='</div>'
s+=verdict("We hoped uncertainty could warn us about the poor. It can&rsquo;t &mdash; their error is <b>bias</b>, not variance. A model can be perfectly calibrated and still <b>silently fail the poorest</b>.")
write_slide("s20b_equity","light",s,css_b)

for n in ["s20a_calibration","s20b_equity"]:
    render(n); print("rendered",n)
