"""Act 6 / Section 05 — the conclusion: divider, cost of targeting, the takeaway, contributions, thanks, references."""
from lib import write_slide, render, header, method_note, verdict
from build_light_deck import GLOBE
CYAN="#16b9d0"; NAVY="#3c5163"; CORAL="#d9747c"
def blk(label, text):
    return f'<div class=blk><div class=bl>{label}</div><div class=bt>{text}</div></div>'

# ---------- S23 · divider 05 ----------
s23_css="""
.gl{position:absolute;inset:0;z-index:0}
.signum{position:absolute;left:118px;top:300px;font-family:'Bricolage Grotesque';font-weight:800;font-size:300px;line-height:.8;color:transparent;background:linear-gradient(150deg,#12b3cc,#1aa07f 60%,#2a6cf0);-webkit-background-clip:text;background-clip:text;z-index:4}
.sigrule{position:absolute;left:134px;top:600px;width:120px;height:6px;border-radius:3px;background:linear-gradient(90deg,#16b9d0,#1aa985);z-index:5}
.sigttl{position:absolute;left:130px;top:640px;font-family:'Bricolage Grotesque';font-weight:800;font-size:74px;color:#0f2636;z-index:5}
.sigsub{position:absolute;left:134px;top:762px;font-size:30px;color:#516470;z-index:5;max-width:1000px;font-weight:500}.sigsub em{font-style:normal;color:#0e8ba0;font-weight:700}
"""
s23=f'<div class=gl>{GLOBE}</div><div class=pg>28 / 33</div><div class=eb>SECTION 05</div>'
s23+='<div class=signum>05</div><div class=sigrule></div><div class=sigttl>So what?</div>'
s23+='<div class=sigsub>What it means for actually <em>using</em> these maps to reach people.</div>'
write_slide("s23_sowhat","light",s23,s23_css)

# ---------- S24 · cost of targeting ----------
s24_css="""
.chart{position:absolute;left:96px;top:300px;width:892px;background:#fff;border:1px solid #dbe5ea;border-radius:16px;padding:26px 30px;box-shadow:0 6px 18px rgba(20,40,55,.06);z-index:5}
.chart .cap{font-family:'JetBrains Mono';font-size:14px;letter-spacing:1.5px;color:#5f7782;margin-bottom:20px}
.rbar{display:flex;height:104px;border-radius:12px;overflow:hidden}
.rbar .seg{display:flex;flex-direction:column;align-items:center;justify-content:center;color:#fff}
.rbar .reach{flex:49;background:linear-gradient(120deg,#16b9d0,#1aa985)}
.rbar .miss{flex:51;background:#d9747c}
.rbar .p{font-family:'Bricolage Grotesque';font-weight:800;font-size:40px}
.rbar .l{font-size:15px;margin-top:2px;opacity:.95}
.note{margin-top:18px;font-size:19px;color:#33454f;line-height:1.4}.note b{color:#0f2636}
.rcol{position:absolute;right:104px;top:300px;width:770px;height:442px;display:flex;flex-direction:column;justify-content:space-between;z-index:5}
.rcol .blk{margin-bottom:0;padding-left:24px}.rcol .bt{font-size:20px;line-height:1.4}
"""
s24=header("If you used it to <em>target aid</em>&hellip;",
           "These maps already steer real money. So we asked the only question that matters in practice: who would you actually reach?",
           eyebrow="05 · SO WHAT?", pageno="29 / 33")
s24+=('<div class=chart><div class=cap>OF THE TRULY-POOREST 20% OF VILLAGES &mdash; WHO TARGETING BY THE MODEL REACHES</div>'
      '<div class=rbar><div class="seg reach"><div class=p>49%</div><div class=l>REACHED</div></div>'
      '<div class="seg miss"><div class=p>51%</div><div class=l>MISSED &mdash; the neediest, skipped</div></div></div>'
      '<div class=note>Tighter still: target only the poorest <b>10%</b> and you reach just <b>39%</b> &mdash; you miss <b>61%</b> of them.</div></div>')
s24+='<div class=rcol>'
s24+=blk("WHAT WE DID","Ranked every village by <b>predicted</b> wealth, took the poorest 20% (who&rsquo;d get aid), and checked how many of the <b>truly</b>-poorest 20% are in it.")
s24+=blk("WHAT WE FOUND","Just <b>49%</b> &mdash; 2.5&times; better than random (20%), but still half the neediest are ranked out of the aid bracket.")
s24+=blk("WHAT IT MEANS","Aid would skew toward the <b>less-poor</b>, and &mdash; from the urban/rural split &mdash; the <b>rural poor</b> are the ones missed most.")
s24+='</div>'
s24+=verdict("Target the poorest with this map and you&rsquo;d miss about <b>half</b> of them &mdash; the rural poor most of all.")
s24+=method_note("Poorest-20% targeting recall computed on the 13,453 held-out villages: overlap between the model&rsquo;s poorest quintile and the true poorest quintile.")
write_slide("s24_targeting","light",s24,s24_css)

# ---------- S25 · the one thing to remember ----------
s25_css="""
.eyb{position:absolute;left:0;top:210px;width:1920px;text-align:center;font-family:'JetBrains Mono';font-size:18px;letter-spacing:6px;color:#1296ad;z-index:5}
.big{position:absolute;left:210px;top:300px;width:1500px;text-align:center;font-family:'Bricolage Grotesque';font-weight:800;font-size:58px;line-height:1.18;color:#0f2636;z-index:5}
.big .good{color:#0e8ba0}.big .bad{color:#cf5b63}
.cards{position:absolute;left:210px;top:660px;width:1500px;display:flex;gap:30px;z-index:5}
.cc{flex:1;border-radius:16px;padding:28px 32px;background:#fff;box-shadow:0 6px 18px rgba(20,40,55,.06)}
.cgood{border-top:5px solid #16b9d0}.cbad{border-top:5px solid #d9747c}
.cc .h{font-family:'Bricolage Grotesque';font-weight:800;font-size:26px;margin-bottom:8px}
.cgood .h{color:#0e8ba0}.cbad .h{color:#cf5b63}
.cc .t{font-size:20px;color:#46606e;line-height:1.4}.cc .t b{color:#0f2636}
"""
s25='<div class=eyb>THE ONE THING TO REMEMBER</div>'
s25+=('<div class=big>Satellite poverty maps are excellent for <span class=good>ranking regions</span><br>'
      '&mdash; and dangerous for <span class=bad>targeting the poorest.</span></div>')
s25+=('<div class=cards>'
      '<div class="cc cgood"><div class=h>&#10003;&nbsp; Trust it to</div><div class=t>Compare whole regions &mdash; <b>which districts are poorer than others</b>. Ranking transfers, even to new countries.</div></div>'
      '<div class="cc cbad"><div class=h>&#10005;&nbsp; Don&rsquo;t trust it to</div><div class=t>Find the neediest villages &mdash; it <b>misses half</b>, is biased against the poorest, and <b>can&rsquo;t tell you when it&rsquo;s wrong</b>.</div></div>'
      '</div>')
s25+='<div class=pg>30 / 33</div>'
write_slide("s25_remember","light",s25,s25_css)

# ---------- S26 · contributions ----------
s26_css="""
.tier{position:absolute;width:1684px;left:118px;z-index:5}
.t1{top:296px}.t2{top:486px}.t3{top:676px}
.tier .row{display:flex;align-items:flex-start;gap:26px}
.tlab{flex:none;width:300px;font-family:'JetBrains Mono';font-size:15px;letter-spacing:2px;padding-top:6px}
.t1 .tlab{color:#7a8d98}.t2 .tlab{color:#0e8ba0}.t3 .tlab{color:#0f2636}
.titems{flex:1;display:flex;gap:22px}
.ti{flex:1;background:#fff;border:1px solid #dbe5ea;border-radius:14px;padding:22px 26px;box-shadow:0 4px 14px rgba(20,40,55,.05)}
.t2 .ti{border-left:4px solid #16b9d0}.t3 .ti{border-left:5px solid #1aa985;background:#f1fbfc}
.ti .h{font-family:'Bricolage Grotesque';font-weight:700;font-size:21px;color:#0f2636;margin-bottom:6px}
.ti .d{font-size:17px;color:#566b78;line-height:1.4}
.hon{position:absolute;left:118px;bottom:74px;width:1684px;font-size:20px;color:#516470;z-index:5}.hon b{color:#0f2636}
"""
def tier(css,label,items):
    s=f'<div class="tier {css}"><div class=row><div class=tlab>{label}</div><div class=titems>'
    for h,d in items: s+=f'<div class=ti><div class=h>{h}</div><div class=d>{d}</div></div>'
    return s+'</div></div></div>'
s26=header("What&rsquo;s <em>new</em> here",
           "An audit, not an architecture &mdash; three layers, from solid foundation to the genuinely novel.",
           eyebrow="05 · SO WHAT?", pageno="31 / 33")
s26+=tier("t1","THE FOUNDATION",[("Faithful replication","Yeh 2020 reproduced at benchmark level on a harder, country-blocked split."),
    ("Night-light dominance","Confirmed the single signal carrying most of the prediction.")])
s26+=tier("t2","THE AUDIT (extends prior work)",[("23-country fairness","Urban&gt;rural ranking gap, scaled from Aiken&rsquo;s 10 countries to 23."),
    ("Decile-resolved bias","The regression-to-mean failure mapped wealth-level by wealth-level."),
    ("Temporal + OOD validity","Holds out future rounds and six entirely unseen countries.")])
s26+=tier("t3","MOST NOVEL",[("Calibration &ne; equity","No uncertainty method flags the poor &mdash; because their error is bias, not variance. An equity-framed UQ negative result."),
    ("The root cause, measured","The dominant signal is physically dark for the poor &mdash; shown, not argued.")])
s26+='<div class=hon><b>Honest framing:</b> this is a rigorous audit and a negative result &mdash; no new state-of-the-art is claimed.</div>'
write_slide("s26_contributions","light",s26,s26_css)

# ---------- S27 · thank you ----------
s27_css="""
.gl{position:absolute;inset:0;z-index:0}
.ty{position:absolute;left:130px;top:330px;font-family:'Bricolage Grotesque';font-weight:800;font-size:104px;color:#0f2636;z-index:5}
.ty .d{color:#16b9d0}
.q{position:absolute;left:134px;top:486px;font-size:32px;color:#516470;z-index:5}.q b{color:#0e8ba0}
.pocket{position:absolute;left:134px;top:586px;width:800px;z-index:5}
.pocket .lab{font-family:'JetBrains Mono';font-size:14px;letter-spacing:2px;color:#7a8d98;margin-bottom:14px}
.chips{display:flex;gap:16px;flex-wrap:wrap}
.chip2{background:#0f2636;border-radius:12px;padding:14px 20px}
.chip2 .n{font-family:'Bricolage Grotesque';font-weight:800;font-size:26px;color:#3fe3f0}
.chip2 .c{font-size:14px;color:#a9bcc8;margin-top:2px}
.who{position:absolute;left:134px;bottom:80px;font-size:20px;color:#16303f;font-weight:600;z-index:5}.who span{color:#7c8d99;font-weight:400}
"""
s27=f'<div class=gl>{GLOBE}</div><div class=pg>32 / 33</div>'
s27+='<div class=ty>Thank you<span class=d>.</span></div><div class=q>Questions &mdash; and a few numbers <b>in my back pocket:</b></div>'
s27+='<div class=pocket><div class=chips>'
for n,c in [("0.76","benchmark Pearson r"),("0.57","r&sup2; leave-country-out"),("+0.62","poorest over-predicted"),("0.28","NL&ndash;wealth, poorest 30%"),("49%","of the neediest reached")]:
    s27+=f'<div class=chip2><div class=n>{n}</div><div class=c>{c}</div></div>'
s27+='</div></div>'
s27+='<div class=who>Onur Haniffa <span>&nbsp;&middot;&nbsp; Advisor: Dr. Seda Nilg&uuml;n Dumlu &nbsp;&middot;&nbsp; Acıbadem MAA&Uuml; &middot; 2026</span></div>'
write_slide("s27_thanks","light",s27,s27_css)

# ---------- S28 · references ----------
s28_css="""
.refs{position:absolute;left:118px;top:296px;width:1684px;columns:2;column-gap:60px;z-index:5}
.ref{break-inside:avoid;margin-bottom:22px;font-size:19px;color:#33454f;line-height:1.4;padding-left:22px;border-left:3px solid #cfe0e6}
.ref b{color:#0f2636}
"""
refs=[("Yeh et al. (2020)","Using publicly available satellite imagery and deep learning to understand economic well-being in Africa. <i>Nature Communications.</i>"),
 ("Koh et al. (2021)","WILDS: A benchmark of in-the-wild distribution shifts (PovertyMap). <i>ICML.</i>"),
 ("Aiken et al. (2022)","Machine learning and phone data to target humanitarian aid (Togo Novissi). <i>Nature.</i>"),
 ("Aiken et al. (2023)","Fairness and representation in satellite-based poverty maps."),
 ("Yang et al. (2021)","Delving into deep imbalanced regression (LDS). <i>ICML.</i>"),
 ("Ren et al. (2022)","Balanced MSE for imbalanced visual regression. <i>CVPR.</i>"),
 ("Rutstein & Johnson (2004)","The DHS Wealth Index. <i>DHS Comparative Reports.</i>"),
 ("Lakshminarayanan et al. (2017)","Simple and scalable predictive uncertainty with deep ensembles. <i>NeurIPS.</i>")]
s28=header("References",None,eyebrow="05 · SO WHAT?",pageno="33 / 33")
s28+='<div class=refs>'
for a,t in refs: s28+=f'<div class=ref><b>{a}</b> &mdash; {t}</div>'
s28+='</div>'
write_slide("s28_references","light",s28,s28_css)

for n in ["s23_sowhat","s24_targeting","s25_remember","s26_contributions","s27_thanks","s28_references"]:
    render(n); print("rendered",n)
