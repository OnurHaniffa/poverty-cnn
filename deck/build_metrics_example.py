"""Reworked slide 16: worked-example-only (the squeeze). Definitions now live on the primer (slide 14)."""
from lib import write_slide, render, header, verdict
def blk(label, text):
    return f'<div class=blk><div class=bl>{label}</div><div class=bt>{text}</div></div>'

rows = [("Kibera","&minus;1.5","&minus;0.9"),("Mathare","&minus;0.7","&minus;0.5"),("Dagoretti","0.1","0.0"),("Karen","0.8","0.6"),("Runda","1.6","1.0")]
css = """
.ex{position:absolute;left:118px;top:302px;width:790px;background:#fff;border:1px solid #dbe5ea;border-radius:16px;padding:26px 32px;box-shadow:0 6px 18px rgba(20,40,55,.06);z-index:5}
.ex .cap{font-family:'JetBrains Mono';font-size:14px;letter-spacing:1.5px;color:#5f7782;margin-bottom:16px}
.tb{width:100%;border-collapse:collapse;font-size:25px}
.tb th{text-align:right;font-family:'JetBrains Mono';font-size:14px;color:#7a8d98;font-weight:500;padding:10px 18px;letter-spacing:1px}
.tb td{text-align:right;padding:11px 18px;color:#33454f;border-top:1px solid #eef2f4}
.tb td:first-child,.tb th:first-child{text-align:left;color:#0f2636;font-weight:600}
.tb .pred{color:#0e8ba0;font-weight:700}
.exr{position:absolute;right:104px;top:300px;width:820px;z-index:5}
.exr .blk{padding-left:24px;margin-bottom:24px}.exr .bt{font-size:21px;line-height:1.5}
"""
s = header("The catch: same order, <em>squeezed values</em>",
           "A model can rank villages <b>perfectly</b> yet get every number <b>wrong</b>. Watch what &lsquo;regression to the mean&rsquo; does &mdash; it&rsquo;s the seed of the whole second half.",
           eyebrow="03 · DOES IT WORK?", pageno="16 / 33")
tbl = '<table class=tb><tr><th>village</th><th>true wealth</th><th>model</th></tr>'
for v,t,p in rows: tbl += f'<tr><td>{v}</td><td>{t}</td><td class=pred>{p}</td></tr>'
tbl += '</table>'
s += f'<div class=ex><div class=cap>SAME ORDER, SQUEEZED VALUES &mdash; FIVE NAIROBI AREAS</div>{tbl}</div>'
s += '<div class=exr>'
s += blk("THE VALUES GET SQUEEZED","The model pulls every guess <b>toward the middle</b> &mdash; the poorest predicted less-poor (&minus;1.5 &rarr; &minus;0.9), the richest less-rich. The numbers are off, so <b>r&sup2; and Pearson suffer</b>.")
s += blk("THE ORDER SURVIVES","But poorest still ranks poorest, richest still richest &mdash; so <b>Spearman stays perfect</b>. That gap is exactly why we lead with <b>ranking</b>.")
s += blk("THE SEED OF THE BIAS","That squeeze toward the mean is <b>regression to the mean</b>. Harmless here &mdash; but on the real data it becomes the <b>staircase</b> that fails the poorest (slide 23).")
s += '</div>'
s += verdict("Perfect ranking, squeezed values &mdash; the first sign of the bias that defines the second half.")
write_slide("s13_metrics", "light", s, css)
render("s13_metrics"); print("rendered reworked s13_metrics")
