"""S2 in the LOCKED light scheme (identical palette to the PCA slide)."""
from lib import write_slide, render, header

css = """
.card{position:absolute;top:252px;width:830px;border-radius:18px;padding:30px 34px;z-index:5;background:#fff;box-shadow:0 6px 22px rgba(20,40,55,.06)}
.cL{left:118px;border:1px solid #d7e6ea;border-top:4px solid #16b9d0}
.cR{left:972px;border:1px solid #ecdcdc;border-top:4px solid #e0727a}
.ctag{font-family:'Bricolage Grotesque';font-weight:800;font-size:25px;margin-bottom:20px;display:flex;align-items:center;gap:13px}
.cL .ctag{color:#0e8ba0}.cR .ctag{color:#cf5b63}
.ic{width:34px;height:34px;border-radius:9px;display:flex;align-items:center;justify-content:center;font-size:20px}
.cL .ic{background:rgba(18,150,173,.12);color:#0e8ba0}
.cR .ic{background:rgba(224,114,122,.12);color:#cf5b63}
.li{display:flex;gap:15px;margin-bottom:19px;align-items:flex-start}
.li .b{flex:none;width:8px;height:8px;border-radius:50%;margin-top:11px}
.cL .li .b{background:#16b9d0}.cR .li .b{background:#e0727a}
.lead{font-family:'Bricolage Grotesque';font-weight:700;font-size:23px;color:#0f2636;line-height:1.2;margin-bottom:3px}
.sub{font-size:19.5px;color:#566b78;line-height:1.42}.sub b{color:#0f2636;font-weight:700}
.sublab{position:absolute;left:118px;top:690px;font-family:'JetBrains Mono';font-size:15px;letter-spacing:3px;color:#1296ad;z-index:5}
.pillars{position:absolute;left:118px;top:728px;width:1684px;display:flex;gap:22px;z-index:5}
.pill{flex:1;background:#fff;border:1px solid #dbe5ea;border-left:5px solid #16b9d0;border-radius:14px;padding:24px 26px;box-shadow:0 4px 14px rgba(20,40,55,.05)}
.pn{font-family:'Bricolage Grotesque';font-weight:800;font-size:26px;color:#0f2636;margin-bottom:9px}
.pn span{color:#1296ad;font-size:20px}
.pq{font-size:20.5px;color:#566b78;line-height:1.42}.pq b{color:#0f2636;font-weight:700}
"""

is_items = [
 ("A faithful replication of Yeh et&nbsp;al. (2020)", "8-band satellite imagery &rarr; a ResNet-18 &rarr; village wealth, on the same data family."),
 ("The audit the original never ran", "We then ask whether that model is <b>fair, calibrated, and robust</b> &mdash; the questions Yeh left open."),
 ("Honest, cross-country evaluation", "Every number is measured on countries the model <b>never trained on</b>, across all 23."),
]
isnt_items = [
 ("Not a new architecture or SOTA claim", "The network is a standard ResNet-18 &mdash; the contribution is the <b>evaluation</b>, not the model."),
 ("Not a deploy-ready targeting tool", "This is a scientific audit, not a system to hand an aid agency tomorrow."),
 ("Not a criticism of the original", "We build on Yeh 2020 and <b>reproduce its headline</b> before stress-testing it."),
]
pillars = [
 ("Fairness", "Is it as accurate for the <b>rural poor</b> as for rich cities?"),
 ("Uncertainty", "When it&rsquo;s wrong, does it at least <b>know it&rsquo;s unsure</b>?"),
 ("Generalization", "Does it still work in countries it has <b>never seen</b>?"),
]

b = header("What this is &mdash; and what it <em>isn&rsquo;t</em>",
           "We reproduce Yeh et&nbsp;al. (2020) faithfully &mdash; then run the audit the original paper didn&rsquo;t.",
           eyebrow="00 · FRAME", pageno="02 / 33")
b += '<div class="card cL"><div class=ctag><span class=ic>&#10003;</span>WHAT IT IS</div>'
for lead, sub in is_items:
    b += f'<div class=li><div class=b></div><div><div class=lead>{lead}</div><div class=sub>{sub}</div></div></div>'
b += '</div><div class="card cR"><div class=ctag><span class=ic>&#10005;</span>WHAT IT ISN&rsquo;T</div>'
for lead, sub in isnt_items:
    b += f'<div class=li><div class=b></div><div><div class=lead>{lead}</div><div class=sub>{sub}</div></div></div>'
b += '</div>'
b += '<div class=sublab>THE THREE QUESTIONS THIS TALK ANSWERS</div><div class=pillars>'
for i,(n,q) in enumerate(pillars):
    b += f'<div class=pill><div class=pn><span>0{i+1}</span>&nbsp; {n}</div><div class=pq>{q}</div></div>'
b += '</div>'
write_slide("s02_frame", "light", b, css)
render("s02_frame"); print("rebuilt s02_frame LIGHT")
