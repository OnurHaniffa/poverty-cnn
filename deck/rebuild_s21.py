"""S21 as a cohesive story: why -> what we did -> what we found -> what it means, + verdict."""
from lib import write_slide, render, header, method_note, verdict
CYAN="#16b9d0"; NAVY="#3c5163"; CORAL="#d9747c"
def blk(label, text):
    return f'<div class=blk><div class=bl>{label}</div><div class=bt>{text}</div></div>'

def fixchart(bars, W=880, H=404):
    n=len(bars); pw=W/n; baseY=H-94; topY=56; bh=baseY-topY; bw=88; vmax=1.5
    ly=baseY-0.618/vmax*bh
    parts=[f'<line x1="40" y1="{ly:.0f}" x2="{W-20}" y2="{ly:.0f}" stroke="#9fb1bd" stroke-width="1.4" stroke-dasharray="6 6"/>',
           f'<text x="42" y="{topY-24}" font-size="14" font-family="Hanken Grotesk" fill="#7a8d98">poorest-decile bias &mdash; <tspan fill="#0f2636" font-weight="700">lower is better</tspan></text>']
    for i,(lab,bias,r2,note,ncol) in enumerate(bars):
        cx=pw*(i+0.5); h=bias/vmax*bh; col=CORAL if bias>0.8 else (NAVY if i==0 else CYAN)
        parts.append(f'<rect x="{cx-bw/2:.0f}" y="{baseY-h:.0f}" width="{bw}" height="{h:.0f}" rx="6" fill="{col}"/>')
        parts.append(f'<text x="{cx:.0f}" y="{baseY-h-11:.0f}" text-anchor="middle" font-size="20" font-weight="700" font-family="Hanken Grotesk" fill="#0f2636">+{bias:.2f}</text>')
        parts.append(f'<text x="{cx:.0f}" y="{baseY+24:.0f}" text-anchor="middle" font-size="16" font-family="Hanken Grotesk" fill="#48606e" font-weight="600">{lab}</text>')
        parts.append(f'<text x="{cx:.0f}" y="{baseY+44:.0f}" text-anchor="middle" font-size="14" font-family="JetBrains Mono" fill="#90a4af">r&sup2; {r2:.2f}</text>')
        parts.append(f'<text x="{cx:.0f}" y="{baseY+66:.0f}" text-anchor="middle" font-size="15" font-style="italic" font-family="Hanken Grotesk" fill="{ncol}">{note}</text>')
    return f'<svg width="{W}" height="{H}" viewBox="0 0 {W} {H}">{"".join(parts)}</svg>'

chart=fixchart([("Standard model",0.618,0.569,"the disease itself",NAVY),
                ("Loss reweighting",0.599,0.535,"barely below",NAVY),
                ("Balanced-MSE",1.411,0.114,"far worse",CORAL)])
css = """
.chart{position:absolute;left:96px;top:300px;width:892px;background:#fff;border:1px solid #dbe5ea;border-radius:16px;padding:22px 24px 10px;box-shadow:0 6px 18px rgba(20,40,55,.06);z-index:5}
.chart .cap{font-family:'JetBrains Mono';font-size:14px;letter-spacing:1.5px;color:#5f7782;margin-bottom:6px}
.rcol{position:absolute;right:104px;top:292px;width:770px;z-index:5}
.rcol .blk{padding-left:24px;margin-bottom:22px}.rcol .bt{font-size:20.5px;line-height:1.42}
"""
s21 = header("We tried to fix it. <em>Nothing works.</em>",
             "Is the poorest-bias a fixable tuning artifact &mdash; or something deeper? We ran the field&rsquo;s standard cures to find out.",
             eyebrow="04 · THE TURN", pageno="26 / 33")
s21 += f'<div class=chart><div class=cap>POOREST-DECILE BIAS &mdash; STANDARD MODEL vs EACH ATTEMPTED FIX</div>{chart}</div>'
s21 += '<div class=rcol>'
s21 += blk("WHAT WE TRIED","Starting from our standard model (the <b>+0.62-biased baseline</b>), we retrained it with the two go-to de-biasing losses &mdash; <b>reweighting</b> (LDS) and <b>Balanced-MSE</b> &mdash; and separately tripled the data.")
s21 += blk("WHAT HAPPENED","Reweighting nudged the bias <b>+0.62 &rarr; +0.60</b> as r&sup2; fell. Balanced-MSE wrecked accuracy (0.57&rarr;0.11) and made it <b>worse</b> (+1.41). More data lifted overall r&sup2; but left the poorest untouched.")
s21 += blk("WHAT IT MEANS","A slope is trivially re-scaled, but no loss can <b>invent ranking signal the imagery lacks</b> among the poor. The limit is the data, not the recipe &mdash; as the next slide shows.")
s21 += '</div>'
s21 += verdict("The <b>level</b> is correctable, but no fix recovers the ability to <b>rank</b> the poorest &mdash; a signal limit, not a tuning problem.")
s21 += method_note("We retrained the baseline with LDS reweighting and a Balanced-MSE objective, then re-measured the same poorest-decile bias. Tripling the data doesn&rsquo;t move it either.")
write_slide("s21_nothing_works", "light", s21, css)
render("s21_nothing_works"); print("rebuilt s21")
