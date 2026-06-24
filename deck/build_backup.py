"""Backup slide: controls & robustness — pre-empts the methods questions."""
from lib import write_slide, render, header, verdict, method_note

css = """
.grid{position:absolute;left:118px;top:300px;width:1684px;display:grid;grid-template-columns:1fr 1fr;gap:28px;z-index:5}
.bc{background:#fff;border:1px solid #dbe5ea;border-left:5px solid #16b9d0;border-radius:16px;padding:28px 32px;box-shadow:0 5px 16px rgba(20,40,55,.05)}
.bc .q{font-family:'JetBrains Mono';font-size:14px;letter-spacing:1.5px;color:#1296ad;margin-bottom:10px}
.bc .h{font-family:'Bricolage Grotesque';font-weight:800;font-size:25px;color:#0f2636;margin-bottom:8px}
.bc .h b{color:#0e8ba0}
.bc .t{font-size:19px;color:#566b78;line-height:1.45}.bc .t b{color:#0f2636}
"""
def card(q,h,t):
    return f'<div class=bc><div class=q>{q}</div><div class=h>{h}</div><div class=t>{t}</div></div>'

s = header("Controls &amp; robustness",
           "The checks behind the headline claims &mdash; ready for the questions.",
           eyebrow="APPENDIX", pageno="BACKUP · 1")
s += '<div class=grid>'
s += card("&ldquo;DOESN&rsquo;T A SIMPLE REGRESSION DO THIS?&rdquo;",
          "Night-lights-only: <b>Spearman 0.76</b>, r&sup2; <b>0.16</b>",
          "A 1-feature night-light model <b>ranks</b> villages as well as the CNN &mdash; but a linear fit captures almost none of the <b>level</b> (r&sup2; 0.16 vs 0.57). The CNN&rsquo;s value is the non-linear level, not the ranking.")
s += card("&ldquo;WOULD A FANCIER NETWORK HELP?&rdquo;",
          "Pretrained: <b>&minus;0.03 r&sup2;</b>",
          "ImageNet pre-training makes it slightly <b>worse</b> than from-scratch. The gap to Yeh isn&rsquo;t the architecture &mdash; <b>it&rsquo;s the data budget</b>. A bigger network won&rsquo;t close it.")
s += card("&ldquo;DO YOUR CV FOLDS LEAK SPATIALLY?&rdquo;",
          "Residual Moran&rsquo;s I <b>0.18&ndash;0.47</b>",
          "Residuals are spatially autocorrelated <b>within</b> countries (expected), but we hold out <b>whole countries</b> &mdash; so cross-fold leakage is limited to border regions, not the bulk of the test set.")
s += card("&ldquo;ISN&rsquo;T A SLOPE OF 0.60 JUST CORRECTABLE?&rdquo;",
          "Yes &mdash; the <b>level</b>, not the <b>ranking</b>",
          "An affine rescale removes the +0.62 bias on paper. But it cannot recover <b>order among the poorest</b> &mdash; because the signal isn&rsquo;t there (slide 27). That&rsquo;s why we call it a signal limit, not a tuning bug.")
s += '</div>'
write_slide("s29_backup_controls", "light", s, css)
render("s29_backup_controls"); print("rendered backup")

# ---------- Backup 2: within-country robustness ----------
wc_css = """
.cards{position:absolute;left:118px;top:300px;width:1684px;display:flex;gap:30px;z-index:5}
.rc{flex:1;background:#fff;border:1px solid #dbe5ea;border-top:5px solid #16b9d0;border-radius:16px;padding:30px 34px;box-shadow:0 5px 16px rgba(20,40,55,.05)}
.rc .lab{font-family:'JetBrains Mono';font-size:14px;letter-spacing:2px;color:#1296ad;margin-bottom:16px}
.rc .big{font-family:'Bricolage Grotesque';font-weight:800;font-size:60px;line-height:1;color:transparent;background:linear-gradient(96deg,#12b3cc,#1aa07f);-webkit-background-clip:text;background-clip:text}
.rc .ctx{font-size:15px;color:#7a8d98;margin:8px 0 16px;font-family:'JetBrains Mono'}
.rc .t{font-size:20px;color:#566b78;line-height:1.45}.rc .t b{color:#0f2636}
"""
def rcard(lab,big,ctx,t):
    return f'<div class=rc><div class=lab>{lab}</div><div class=big>{big}</div><div class=ctx>{ctx}</div><div class=t>{t}</div></div>'
w=header("Robustness: it survives <em>within-country</em>",
         "The wealth index is pooled across 23 countries &mdash; so are the findings just the model sorting rich countries from poor? We removed the between-country signal and re-checked every claim.",
         eyebrow="APPENDIX", pageno="BACKUP · 2")
w+='<div class=cards>'
w+=rcard("RANKING HOLDS","0.70","within-country Spearman (vs 0.72 pooled)","It ranks villages <b>inside</b> a country almost as well as overall &mdash; it isn&rsquo;t just sorting countries.")
w+=rcard("SHRINKAGE IS REAL","0.65","within-country slope &middot; 23/23 countries","Regression to the mean (slope &lt;&nbsp;1) shows up <b>within every country</b> &mdash; not a between-country effect.")
w+=rcard("POOREST-BIAS IS REAL","+0.41","poorest-20% bias &middot; 21/23 countries","It over-predicts the poorest villages <b>inside each country</b> too &mdash; the failure is genuinely local.")
w+='</div>'
w+=verdict("All three findings survive removing the between-country signal &mdash; they&rsquo;re real <b>within-country</b>, not artifacts of the pooled index.")
w+=method_note("Each claim re-evaluated against a within-country-standardised wealth index (between-country level removed), per country, then n-weighted across all 23 countries.")
write_slide("s30_backup_within","light",w,wc_css)
render("s30_backup_within"); print("rendered within-country backup")
