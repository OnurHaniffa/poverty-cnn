"""S12 rebuilt: balanced 3-card row with r2 bars, on-palette (grey/cyan/navy), space-filling."""
from lib import write_slide, render, header

css = """
.row{position:absolute;left:118px;top:292px;width:1684px;display:flex;gap:30px;z-index:5}
.card{flex:1;background:#fff;border:1px solid #dde6ea;border-radius:18px;padding:32px 34px;display:flex;flex-direction:column;min-height:496px;box-shadow:0 5px 16px rgba(20,40,55,.05)}
.card.hero{border:2px solid #16b9d0;box-shadow:0 14px 32px rgba(22,185,208,.20)}
.reg{font-family:'JetBrains Mono';font-size:15px;letter-spacing:2px;color:#7a8d98;margin-bottom:16px}
.hero .reg{color:#0e8ba0}
.r2{font-family:'Bricolage Grotesque';font-weight:800;font-size:74px;line-height:1}
.c1 .r2{color:#9aa7af}.hero .r2{color:transparent;background:linear-gradient(96deg,#12b3cc,#1aa07f);-webkit-background-clip:text;background-clip:text}.c3 .r2{color:#3c5163}
.r2 span{font-size:26px;color:#9fb1bd;font-family:'Hanken Grotesk';font-weight:600}
.track{height:15px;border-radius:8px;background:#eef2f4;margin:20px 0 8px}
.fill{height:100%;border-radius:8px}
.c1 .fill{background:#c3ced4}.hero .fill{background:linear-gradient(90deg,#16b9d0,#1aa985)}.c3 .fill{background:#5b7384}
.sub20{font-size:15px;color:#0e8ba0;font-weight:700;margin-bottom:14px}
.mean{font-size:20.5px;color:#566b78;line-height:1.46;flex:1}.mean b{color:#0f2636}
.badge{align-self:flex-start;margin-top:18px;font-family:'JetBrains Mono';font-size:13px;letter-spacing:1px;padding:8px 14px;border-radius:20px}
.bad{background:#eef2f4;color:#86939b}.good{background:rgba(22,185,208,.16);color:#0e8ba0}.neu{background:#eef2f4;color:#5b7384}
.axis{position:absolute;left:118px;top:820px;width:1684px;text-align:center;font-family:'JetBrains Mono';font-size:14px;letter-spacing:2px;color:#90a4af;z-index:5}
.note{position:absolute;left:118px;bottom:56px;width:1684px;text-align:center;font-size:23px;color:#33454f;z-index:5}.note b{color:#0e8ba0}
"""
s12 = header("Why a random split <em>lies</em> &mdash; and how we test honestly",
             "The more honest the test, the lower the score. We climb the ladder and report the honest rung, not the flattering one.",
             eyebrow="03 · DOES IT WORK?", pageno="15 / 33")
s12 += '<div class=row>'
s12 += ('<div class="card c1"><div class=reg>RANDOM SPLIT</div><div class=r2>0.73<span> r&sup2;</span></div>'
        '<div class=track><div class=fill style="width:86%"></div></div>'
        '<div class=mean>Shuffle all villages, then test on held-out ones &mdash; the model can <b>memorise each country</b>. Flattering, but fake.</div>'
        '<div class="badge bad">&#10005;&nbsp; inflated &mdash; we don&rsquo;t use this</div></div>')
s12 += ('<div class="card hero"><div class=reg>LEAVE-COUNTRY-OUT</div><div class=r2>0.57<span> r&sup2;</span></div>'
        '<div class=track><div class=fill style="width:72%"></div></div>'
        '<div class=sub20>&darr; the honest number &mdash; a random split would inflate it to 0.73</div>'
        '<div class=mean>Hold out <b>whole countries</b> &mdash; test on places the model never saw, with country-bootstrap 95% CIs. <b>Every number in this talk is measured this way.</b></div>'
        '<div class="badge good">&#10003;&nbsp; what we report</div></div>')
s12 += ('<div class="card c3"><div class=reg>OUT-OF-DISTRIBUTION</div><div class=r2>0.48<span> r&sup2;</span></div>'
        '<div class=track><div class=fill style="width:56%"></div></div>'
        '<div class=mean>Brand-new countries &mdash; this <b>pooled</b> r&sup2; hides a per-country collapse on the rich tail (slide 20).</div>'
        '<div class="badge neu">hardest test</div></div>')
s12 += '</div>'
s12 += '<div class=axis>EASIEST / MOST FLATTERING &nbsp;&mdash;&mdash;&mdash;&mdash;&mdash;&rarr;&nbsp; HARDEST / MOST HONEST</div>'
s12 += '<div class=note>Analogy: a random split is an exam with questions you <b>studied</b>; leave-country-out is questions you&rsquo;ve <b>never seen</b>. We grade ourselves on the second one.</div>'
write_slide("s12_cvladder", "light", s12, css)
render("s12_cvladder"); print("rebuilt s12")
