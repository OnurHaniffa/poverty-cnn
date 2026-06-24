"""S4 + S5 rebuilt to FILL the vertical space (flex space-between) with larger, readable type."""
from lib import write_slide, render, header
from build_light_deck import GLOBE  # reuse the light globe import side-effects safe

def blk(label, text):
    return f'<div class=blk><div class=bl>{label}</div><div class=bt>{text}</div></div>'

# ---------- S4 · PROBLEM (full-height, two balanced columns) ----------
s4_css = """
.cols{position:absolute;left:118px;top:286px;width:1684px;height:628px;display:flex;z-index:5}
.lcol{flex:1.25;display:flex;flex-direction:column;justify-content:space-between;padding-right:74px}
.vd{width:1px;background:linear-gradient(180deg,transparent,#cdd9df 18%,#cdd9df 82%,transparent)}
.rcol{width:600px;display:flex;flex-direction:column;justify-content:space-between;padding-left:68px}
.cols .blk{margin-bottom:0;padding-left:24px}
.cols .bl{font-size:16px;letter-spacing:2.5px;margin-bottom:9px}
.cols .bt{font-size:26px;line-height:1.5}
.su .sl{font-family:'JetBrains Mono';font-size:16px;letter-spacing:3px;color:#1296ad;margin-bottom:12px}
.su .bsn{font-family:'Bricolage Grotesque';font-weight:800;font-size:128px;line-height:.92;color:transparent;background:linear-gradient(96deg,#12b3cc,#1aa07f);-webkit-background-clip:text;background-clip:text}
.su .bsc{font-size:25px;color:#566b78;margin-top:10px;line-height:1.4}.su .bsc b{color:#0f2636}
"""
s4 = header("Finding the poor is slow, costly, and <em>out of date</em>",
            "To help people in poverty, you first have to <b>find</b> them &mdash; and the ground-truth data to do that is scarce.",
            eyebrow="01 · THE PROMISE", pageno="04 / 33")
s4 += '<div class=cols><div class=lcol>'
s4 += blk("WHY IT'S HARD","Household surveys are the gold standard, but they&rsquo;re <b>expensive, slow, and sparse</b> &mdash; many regions are surveyed only once every several years, if at all.")
s4 += blk("THE OPPORTUNITY","Satellites image <b>the entire planet, for free, repeatedly</b>. If a model can read wealth from that imagery, it becomes a scalable <b>complement</b> to surveys &mdash; filling the gaps in time and space.")
s4 += blk("THE STAKES","These maps already steer <b>real aid</b>: Togo&rsquo;s Novissi program used satellite poverty maps to target emergency COVID cash (Aiken&nbsp;2022). When the map is wrong about the poorest, real people are missed.")
s4 += '</div><div class=vd></div><div class=rcol>'
s4 += ('<div class=su><div class=sl>THE NEED</div><div class=bsn>1.8B</div>'
       '<div class=bsc>people below the poverty line &mdash; and the data to <b>locate</b> them is scarce and stale</div></div>')
s4 += ('<div class=su><div class=sl>THE DATA GAP</div><div class=bsn>~5 yrs</div>'
       '<div class=bsc>typical gap between household surveys &mdash; <b>often far longer</b> in the poorest places</div></div>')
s4 += '</div></div>'
write_slide("s04_problem", "light", s4, s4_css)

# ---------- S5 · IDEA (taller pipeline + two bottom blocks fill the space) ----------
s5_css = """
.pipe{position:absolute;left:118px;top:300px;width:1684px;display:flex;justify-content:space-between;z-index:5}
.step{width:300px;background:#fff;border:1px solid #d8e3e8;border-top:4px solid #16b9d0;border-radius:16px;padding:30px 26px;box-shadow:0 6px 18px rgba(20,40,55,.06);min-height:240px}
.bdg{width:52px;height:52px;border-radius:13px;display:flex;align-items:center;justify-content:center;font-family:'Bricolage Grotesque';font-weight:800;font-size:25px;color:#fff;background:linear-gradient(120deg,#16b9d0,#1aa985);margin-bottom:20px}
.stt{font-family:'Bricolage Grotesque';font-weight:700;font-size:26px;color:#0f2636;margin-bottom:11px}
.sds{font-size:21px;color:#566b78;line-height:1.45}.sds b{color:#0f2636}
.conn{position:absolute;top:425px;height:2px;background:repeating-linear-gradient(90deg,#9fd6e0 0 10px,transparent 10px 20px);z-index:4}
.bottom2{position:absolute;left:118px;top:596px;width:1684px;display:flex;gap:30px;z-index:5}
.bcard{flex:1;background:#fff;border:1px solid #dbe5ea;border-left:5px solid #16b9d0;border-radius:16px;padding:30px 34px;box-shadow:0 5px 16px rgba(20,40,55,.05)}
.bcl{font-family:'JetBrains Mono';font-size:15px;letter-spacing:2.5px;color:#1296ad;margin-bottom:12px}
.bct{font-size:25px;line-height:1.5;color:#33454f}.bct b{color:#0f2636;font-weight:700}
"""
steps=[("1","GPS","Every surveyed village comes with a GPS coordinate."),
       ("2","Satellite tile","We pull a 6.7&nbsp;km image centred on that point."),
       ("3","8 bands","Colour &middot; infrared &middot; thermal &middot; <b>night-lights</b> &mdash; eight ways to see it."),
       ("4","ResNet-18","A standard CNN reads all eight channels together."),
       ("5","One number","It outputs a single value: the predicted <b>wealth</b>.")]
s5 = header("The idea: <em>one model, one number</em>",
            "The whole system in one line &mdash; then we spend the rest of the talk unpacking every piece honestly.",
            eyebrow="01 · THE PROMISE", pageno="05 / 33")
s5 += '<div class=pipe>'
for n,t,d in steps:
    s5 += f'<div class=step><div class=bdg>{n}</div><div class=stt>{t}</div><div class=sds>{d}</div></div>'
s5 += '</div>'
s5 += '<div class=bottom2>'
s5 += ('<div class=bcard><div class=bcl>WHY IT CAN WORK</div><div class=bct>Wealth has a <b>physical footprint</b> visible from above &mdash; dense metal roofs, paved roads, and electric light at night. The CNN learns to read those cues into a single score.</div></div>')
s5 += ('<div class=bcard><div class=bcl>WHAT IT LEARNS FROM</div><div class=bct>Every prediction is <b>trained to match the DHS wealth index</b> &mdash; the &ldquo;answer key&rdquo; we build with PCA in the next section. No survey label, no training signal.</div></div>')
s5 += '</div>'
write_slide("s05_idea", "light", s5, s5_css)

for n in ["s04_problem","s05_idea"]:
    render(n); print("rendered", n)
