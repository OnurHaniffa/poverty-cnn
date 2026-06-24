"""Metrics primer — opens section 3: plain-English Pearson / r2 / Spearman before any results."""
from lib import write_slide, render, header, verdict

css = """
.cards{position:absolute;left:118px;top:284px;width:1684px;display:flex;gap:30px;z-index:5}
.mc{flex:1;background:#fff;border:1px solid #dbe5ea;border-top:5px solid #16b9d0;border-radius:18px;padding:32px 34px;box-shadow:0 6px 18px rgba(20,40,55,.06)}
.mc .tag{font-family:'JetBrains Mono';font-size:17px;letter-spacing:2px;color:#1296ad;margin-bottom:14px}
.mc .q{font-family:'Bricolage Grotesque';font-weight:800;font-size:27px;line-height:1.2;color:#0f2636;margin-bottom:16px}
.mc .q em{font-style:normal;color:#0e8ba0}
.mc .ex{font-size:18.5px;color:#42555f;line-height:1.52;margin-bottom:18px}.mc .ex b{color:#0f2636}
.mc .r{font-size:15px;font-family:'JetBrains Mono';color:#7a8d98;border-top:1px solid #e7edf0;padding-top:12px}.mc .r b{color:#0f2636}
"""
def mc(tag,q,ex,r): return f'<div class=mc><div class=tag>{tag}</div><div class=q>{q}</div><div class=ex>{ex}</div><div class=r>{r}</div></div>'

s = header("Three numbers, three questions",
           "Before any results &mdash; the three ways we grade the model, in plain English.",
           eyebrow="03 · DOES IT WORK?", pageno="14 / 33")
s += '<div class=cards>'
s += mc("PEARSON r","Do predictions <em>track</em> the truth?",
        "Plot predicted against true wealth: how tightly do the dots hug a <b>straight line</b>? A clean line scores high, a fuzzy cloud scores low &mdash; it&rsquo;s the strength of the linear trend.",
        "&minus;1 &rarr; +1 &middot; <b>the metric the benchmarks report</b>")
s += mc("R&sup2;","How much do we <em>explain</em>?",
        "Of all the ways villages differ in wealth, what <b>share</b> does the model capture? <b>0</b> = no better than guessing the average for everyone; <b>1</b> = perfect. Hit hardest by big misses at the extremes.",
        "0 &rarr; 1 &middot; <b>0.57 = we explain 57%</b>")
s += mc("SPEARMAN &rho;","Did we get the <em>order</em> right?",
        "Forget the exact numbers &mdash; line the villages up poorest&rarr;richest by truth, and by prediction. Do the two line-ups <b>agree</b>? Squeeze or stretch the values all you like; only the order counts.",
        "&minus;1 &rarr; +1 &middot; <b>what targeting actually needs</b>")
s += '</div>'
s += verdict("Pearson and r&sup2; grade the <b>values</b>; Spearman grades the <b>order</b> &mdash; and for finding the poor, the order is what matters.")
write_slide("s12c_metrics_primer", "light", s, css)
render("s12c_metrics_primer"); print("rendered enriched primer")
