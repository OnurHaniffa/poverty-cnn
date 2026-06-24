"""Act 2 EDA pair: S7A cleaning+missingness, S7B the box plots. Real figures, deck palette."""
import base64, pathlib
from lib import write_slide, render, header
DECK = pathlib.Path(__file__).parent
def img(name):
    b = (DECK/"assets/eda"/name).read_bytes()
    return "data:image/png;base64," + base64.b64encode(b).decode()
def blk(label, text):
    return f'<div class=blk><div class=bl>{label}</div><div class=bt>{text}</div></div>'

MISS = img("missing_hist.png"); COUNTRY = img("box_country.png"); UR = img("box_urbanrural.png")

# ---------- S7A · EDA cleaning + missingness ----------
s7a_css = """
.lcol{position:absolute;left:118px;top:284px;width:740px;height:660px;display:flex;flex-direction:column;justify-content:space-between;z-index:5}
.lcol .blk{margin-bottom:0;padding-left:24px}.lcol .bl{font-size:15px;letter-spacing:2.5px;margin-bottom:8px}.lcol .bt{font-size:23px;line-height:1.46}
.wcard{position:absolute;right:104px;top:330px;width:840px;background:#fff;border:1px solid #dbe5ea;border-radius:16px;padding:28px 32px;box-shadow:0 6px 18px rgba(20,40,55,.06);z-index:5}
.wtitle{font-family:'JetBrains Mono';font-size:14px;letter-spacing:1.5px;color:#5f7782;margin-bottom:20px}
.qbar{display:flex;height:66px;border-radius:12px;overflow:hidden}
.seg.a{flex:95;background:linear-gradient(120deg,#16b9d0,#1aa985);display:flex;align-items:center;justify-content:center;color:#fff;font-family:'Bricolage Grotesque';font-weight:800;font-size:26px}
.seg.b{flex:5;background:#cfdae0}
.legend{display:flex;gap:36px;margin-top:20px}
.lg{display:flex;align-items:flex-start;gap:11px;font-size:19px;color:#33454f;line-height:1.35}
.lg .d{width:15px;height:15px;border-radius:4px;margin-top:3px;flex:none}.lg b{color:#0f2636}
.qstrip{position:absolute;right:104px;top:648px;width:840px;display:flex;gap:16px;z-index:5}
.takeaway{position:absolute;right:104px;top:792px;width:840px;font-size:21px;color:#0e7c86;font-weight:600;z-index:5;line-height:1.4}.takeaway b{color:#0f2636}
.qs{flex:1;background:#0f2636;color:#eaf4fa;border-radius:13px;padding:16px 20px;text-align:center}
.qs .n{font-family:'Bricolage Grotesque';font-weight:800;font-size:34px;color:#3fe3f0}
.qs .c{font-size:14.5px;color:#a9bcc8;margin-top:3px}
"""
s7a = header("First we clean the data &mdash; then prove <em>how clean</em> it is",
             "Every modelling result rests on the data underneath it, so the cleaning is explicit &mdash; and measurable.",
             eyebrow="02 · THE INGREDIENTS", pageno="08 / 33")
s7a += '<div class=lcol>'
s7a += blk("MISSING VALUES","DHS &ldquo;don&rsquo;t know&rdquo; codes (96&ndash;99) become NaN, then imputed by the <b>within-country mean</b> &mdash; a missing answer never reads as a real zero.")
s7a += blk("COVERAGE FILTER","We drop households missing more than <b>30%</b> of their asset fields &mdash; too little signal to place them reliably.")
s7a += blk("GPS QUALITY","Clusters at &ldquo;null island&rdquo; (0,&nbsp;0) or with broken coordinates are <b>removed</b> before any imagery is matched.")
s7a += blk("THE IMAGERY","Any cloud or missing pixel in a tile is <b>mean-filled per channel</b> &mdash; the network never sees a hole.")
s7a += '</div>'
s7a += ('<div class=wcard><div class=wtitle>HOW COMPLETE IS EACH SATELLITE TILE?</div>'
        '<div class=qbar><div class="seg a">95%</div><div class="seg b"></div></div>'
        '<div class=legend>'
        '<div class=lg><div class=d style="background:linear-gradient(120deg,#16b9d0,#1aa985)"></div><div><b>95% of tiles</b> &mdash; under 5% missing pixels (essentially complete)</div></div>'
        '<div class=lg><div class=d style="background:#cfdae0"></div><div><b>5% of tiles</b> &mdash; 5&ndash;19% missing, mean-filled per channel</div></div>'
        '</div></div>')
s7a += ('<div class=qstrip>'
        '<div class=qs><div class=n>0.66%</div><div class=c>mean missing per tile</div></div>'
        '<div class=qs><div class=n>95%</div><div class=c>of tiles under 5% missing</div></div>'
        '<div class=qs><div class=n>19%</div><div class=c>worst single tile (still 81% intact)</div></div></div>')
s7a += '<div class=takeaway>&rarr; So the gap-filling is negligible &mdash; the network trains on <b>genuine imagery, not invented pixels.</b></div>'
write_slide("s07a_eda_clean", "light", s7a, s7a_css)

# ---------- S7B · EDA the box plots ----------
s7b_css = """
.cfig{position:absolute;left:108px;top:268px;width:1230px;background:#fff;border:1px solid #dbe5ea;border-radius:16px;padding:18px 22px 10px;box-shadow:0 6px 18px rgba(20,40,55,.06);z-index:5}
.cfig img{width:100%;display:block}.cfig .cap{font-family:'JetBrains Mono';font-size:14px;letter-spacing:1.5px;color:#5f7782;margin-bottom:4px}
.rblk{position:absolute;right:104px;top:300px;width:430px;z-index:5}
.rblk .blk{padding-left:24px}.rblk .bt{font-size:22px;line-height:1.46}.rblk .bl{font-size:15px;letter-spacing:2.5px;margin-bottom:8px}
.ufig{position:absolute;left:108px;top:740px;width:330px;background:#fff;border:1px solid #dbe5ea;border-radius:16px;padding:16px 18px 8px;box-shadow:0 6px 18px rgba(20,40,55,.06);z-index:5}
.ufig img{width:100%;display:block}.ufig .cap{font-family:'JetBrains Mono';font-size:13px;letter-spacing:1.5px;color:#5f7782;margin-bottom:4px}
.split{position:absolute;left:480px;top:756px;width:870px;z-index:5}
.split .blk{padding-left:24px}.split .bt{font-size:23px;line-height:1.5}.split .bl{font-size:15px;letter-spacing:2.5px;margin-bottom:8px}
.bignum{font-family:'Bricolage Grotesque';font-weight:800;color:#0e8ba0}
"""
s7b = header("The data at a glance",
             "Box plots show the <b>spread</b> of village wealth &mdash; and the one split that drives the second half of this talk.",
             eyebrow="02 · THE INGREDIENTS", pageno="10 / 33")
s7b += f'<div class=cfig><div class=cap>WEALTH INDEX BY COUNTRY &mdash; POOREST (DRC) &rarr; RICHEST (GHANA)</div><img src="{COUNTRY}"></div>'
s7b += '<div class=rblk>'
s7b += blk("HOW TO READ IT","Each box = the middle 50% of villages; the line is the median; whiskers span the range; dots are outliers.")
s7b += blk("WHY IT MATTERS","Wealth varies <b>within</b> every country, with heavy overlap between them &mdash; so this is <b>not</b> a trivial &lsquo;guess the country&rsquo; task.")
s7b += '</div>'
s7b += f'<div class=ufig><div class=cap>URBAN vs RURAL</div><img src="{UR}"></div>'
s7b += '<div class=split>'
s7b += blk("THE SPLIT THAT MATTERS","Urban villages sit far above rural ones &mdash; median <span class=bignum>+0.91</span> vs <span class=bignum>&minus;0.51</span>. Rural wealth is bunched near the bottom. This urban&ndash;rural gap is the thread we pull on in the fairness section &mdash; the model will turn out to serve cities better than villages.")
s7b += '</div>'
write_slide("s07b_eda_box", "light", s7b, s7b_css)

for n in ["s07a_eda_clean","s07b_eda_box"]:
    render(n); print("rendered", n)
