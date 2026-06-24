"""Act 4: S14 benchmark results, S15 night-lights signal, S16 OOD ranking transfers."""
from lib import write_slide, render, header, method_note
def blk(label, text):
    return f'<div class=blk><div class=bl>{label}</div><div class=bt>{text}</div></div>'

def hbarchart(rows, vmax, ref, ref_lab, W=860, H=470, lbl=270):
    """rows: list of (label, value, valstr, color, highlight_bool)."""
    x0=lbl+18; x1=W-40; rh=(H-30)/len(rows)
    parts=[]
    if ref is not None:
        rx=x0+ref/vmax*(x1-x0)
        parts.append(f'<line x1="{rx:.0f}" y1="6" x2="{rx:.0f}" y2="{H-26:.0f}" stroke="#9fb1bd" stroke-width="1.5" stroke-dasharray="6 6"/>')
        parts.append(f'<text x="{rx:.0f}" y="{H-8}" text-anchor="middle" font-size="13" fill="#7a8d98" font-family="JetBrains Mono">{ref_lab}</text>')
    for i,(label,val,valstr,color,hl) in enumerate(rows):
        cy=14+i*rh+rh/2; bw=val/vmax*(x1-x0); bh=min(rh*0.56,34)
        parts.append(f'<text x="{lbl}" y="{cy+6:.0f}" text-anchor="end" font-size="19" font-family="Hanken Grotesk" font-weight="{700 if hl else 600}" fill="{"#0f2636" if hl else "#48606e"}">{label}</text>')
        parts.append(f'<rect x="{x0}" y="{cy-bh/2:.0f}" width="{bw:.0f}" height="{bh:.0f}" rx="6" fill="{color}"/>')
        parts.append(f'<text x="{x0+bw+12:.0f}" y="{cy+6:.0f}" font-size="18" font-family="Hanken Grotesk" font-weight="700" fill="{"#0e8ba0" if hl else "#7a8d98"}">{valstr}</text>')
    return f'<svg width="{W}" height="{H}" viewBox="0 0 {W} {H}">{"".join(parts)}</svg>'

CY="linear-gradient";  # placeholder
GREY="#c3ced4"; CYAN="#16b9d0"; NAVY="#3c5163"

# ---------- S14 · benchmark ----------
s14_css = """
.stats{position:absolute;left:118px;top:292px;width:1684px;display:flex;gap:30px;z-index:5}
.sc{flex:1;background:#fff;border:1px solid #dbe5ea;border-radius:18px;padding:36px 38px;box-shadow:0 6px 18px rgba(20,40,55,.06)}
.sc.hero{border:2px solid #16b9d0;box-shadow:0 14px 32px rgba(22,185,208,.2)}
.sn{font-family:'Bricolage Grotesque';font-weight:800;font-size:96px;line-height:1;color:transparent;background:linear-gradient(96deg,#12b3cc,#1aa07f);-webkit-background-clip:text;background-clip:text}
.sc:not(.hero) .sn{background:none;color:#0f2636}
.smetric{font-family:'JetBrains Mono';font-size:15px;letter-spacing:2px;color:#1296ad;margin:10px 0 4px}
.sci{font-family:'JetBrains Mono';font-size:14px;color:#90a4af;margin-bottom:12px}
.sd{font-size:21px;color:#566b78;line-height:1.45}.sd b{color:#0f2636}
.blocks{position:absolute;left:118px;top:712px;width:1684px;display:flex;gap:30px;z-index:5}
.blocks .blk{flex:1;padding-left:24px}.blocks .bt{font-size:22px;line-height:1.48}
"""
s14 = header("Does it work? <em>Yes &mdash; at benchmark level</em>",
             "On countries the model never trained on, it performs in the range of published satellite-poverty models.",
             eyebrow="03 · DOES IT WORK?", pageno="18 / 33")
s14 += '<div class=stats>'
s14 += '<div class="sc hero"><div class=sn>0.76</div><div class=smetric>PEARSON r</div><div class=sci>95% CI&nbsp; 0.72&ndash;0.79</div><div class=sd>Correlation with true wealth &mdash; <b>WILDS PovertyMap benchmark territory</b>, on a harder split.</div></div>'
s14 += '<div class=sc><div class=sn>0.57</div><div class=smetric>R&sup2;</div><div class=sci>95% CI&nbsp; 0.51&ndash;0.62</div><div class=sd>Share of wealth variation explained, <b>leave-country-out</b>.</div></div>'
s14 += '<div class=sc><div class=sn>0.72</div><div class=smetric>SPEARMAN &rho;</div><div class=sci>95% CI&nbsp; 0.67&ndash;0.76</div><div class=sd>Ranking &mdash; <b>the metric that matters</b> for targeting.</div></div>'
s14 += '</div>'
s14 += '<div class=blocks>'
s14 += blk("ON UNSEEN COUNTRIES","Every figure is <b>leave-country-out</b> with 95% CIs from a <b>country-level bootstrap</b> &mdash; the test countries were held out of training entirely.")
s14 += blk("HONEST, NOT INFLATED","A random split would have shown r&sup2; <b>0.73</b>. We report the country-blocked <b>0.57</b> &mdash; the inflation we refuse.")
s14 += blk("THE HALF THAT WORKS","This is the &lsquo;it works&rsquo; half of the story. The cracks &mdash; who it works <em>for</em> &mdash; come next.")
s14 += '</div>'
s14 += method_note("Five-fold <b>leave-country-out</b> cross-validation: train on ~18 countries, test on the 5 held out, rotate until every country is tested once &mdash; then pool all 13,453 predictions. These are those pooled numbers.")
write_slide("s14_benchmark", "light", s14, s14_css)

# ---------- S15 · night-lights ----------
abl = [("without Night-lights",0.534,"0.53",CYAN,True),
       ("without RGB colour",0.687,"0.69",GREY,False),
       ("without Near-infrared",0.707,"0.71",GREY,False),
       ("without Thermal",0.719,"0.72",GREY,False),
       ("without Shortwave IR",0.728,"0.73",GREY,False)]
chart15 = hbarchart(abl, 0.80, 0.726, "full model · 0.73", W=900, H=430, lbl=300)
s15_css = """
.chart{position:absolute;left:96px;top:300px;width:920px;background:#fff;border:1px solid #dbe5ea;border-radius:16px;padding:22px 24px 10px;box-shadow:0 6px 18px rgba(20,40,55,.06);z-index:5}
.chart .cap{font-family:'JetBrains Mono';font-size:14px;letter-spacing:1.5px;color:#5f7782;margin-bottom:6px}
.rcol{position:absolute;right:104px;top:312px;width:760px;z-index:5}
.rcol .blk{padding-left:24px}.rcol .bt{font-size:23px;line-height:1.5}
"""
s15 = header("What carries the signal? <em>Night-lights.</em>",
             "Drop any single channel and ranking barely flinches &mdash; except one.",
             eyebrow="03 · DOES IT WORK?", pageno="19 / 33")
s15 += f'<div class=chart><div class=cap>RANKING (SPEARMAN) WHEN EACH CHANNEL IS REMOVED</div>{chart15}</div>'
s15 += '<div class=rcol>'
s15 += blk("ONE CHANNEL DOMINATES","Remove colour, infrared, thermal or SWIR and Spearman stays ~0.70&ndash;0.73. Remove <b>night-lights</b> and it <b>collapses to 0.53</b> &mdash; far more than any other band.")
s15 += blk("WHY: ELECTRIFICATION","Lights at night are a direct proxy for <b>electricity access</b> &mdash; the strongest single signal of village wealth visible from space.")
s15 += blk("THE FORESHADOW","If the model leans this hard on lights &mdash; what happens in the places that have <b>none</b>? Hold that thought.")
s15 += '</div>'
s15 += method_note("We <b>retrained the model once per channel group</b>, each time deleting that group from the 8-band input, and measured how far ranking fell. Only night-lights changed the answer.")
write_slide("s15_nightlights", "light", s15, s15_css)

# ---------- S16 · OOD ----------
ood = [("Gabon",0.813,"0.81",CYAN,True),
       ("Eswatini",0.720,"0.72",NAVY,False),
       ("Niger",0.711,"0.71",NAVY,False),
       ("Madagascar",0.662,"0.66",NAVY,False),
       ("Namibia",0.649,"0.65",NAVY,False),
       ("South Africa",0.484,"0.48",GREY,False)]
chart16 = hbarchart(ood, 0.90, 0.72, "in-distribution · 0.72", W=900, H=470, lbl=270)
s16_css = """
.chart{position:absolute;left:96px;top:296px;width:920px;background:#fff;border:1px solid #dbe5ea;border-radius:16px;padding:22px 24px 10px;box-shadow:0 6px 18px rgba(20,40,55,.06);z-index:5}
.chart .cap{font-family:'JetBrains Mono';font-size:14px;letter-spacing:1.5px;color:#5f7782;margin-bottom:6px}
.rcol{position:absolute;right:104px;top:312px;width:760px;z-index:5}
.rcol .blk{padding-left:24px}.rcol .bt{font-size:23px;line-height:1.5}
"""
s16 = header("It even generalises to <em>countries it never saw</em>",
             "Frozen model, six brand-new countries outside the 23. Ranking transfers &mdash; but watch the extremes.",
             eyebrow="03 · DOES IT WORK?", pageno="20 / 33")
s16 += f'<div class=chart><div class=cap>RANKING (SPEARMAN) ON 6 UNSEEN COUNTRIES</div>{chart16}</div>'
s16 += '<div class=rcol>'
s16 += blk("RANKING TRANSFERS","On countries entirely outside training, Spearman holds at 0.48&ndash;0.81. <b>Gabon (0.81) beats its in-distribution home (0.72)</b> &mdash; the ordering genuinely carries over.")
s16 += blk("BUT r&sup2; BREAKS","On the <b>wealthy extremes</b> the level falls apart: South Africa r&sup2; <b>&minus;1.7</b>, Eswatini <b>&minus;0.6</b>. It can rank villages but can&rsquo;t place where they sit.")
s16 += blk("THE FIRST CRACK","Good for ranking, dangerous for absolute level &mdash; the exact split we dissect in the next act.")
s16 += '</div>'
s16 += method_note("We <b>froze the trained 23-country model</b> and ran it cold on six new countries &mdash; <b>no retraining, and not one survey point from them</b>. Zero data linkage, so this is genuine transfer.")
write_slide("s16_ood", "light", s16, s16_css)

for n in ["s14_benchmark","s15_nightlights","s16_ood"]:
    render(n); print("rendered", n)
