"""Build four original, standalone SVG concepts for WCM's authentication page."""
from pathlib import Path
from math import sin, cos, exp, pi
from xml.etree import ElementTree as ET

OUT = Path(__file__).parent


def svg(name, title, defs, body):
    text = f'''<svg xmlns="http://www.w3.org/2000/svg" width="1920" height="1080" viewBox="0 0 1920 1080" role="img" aria-labelledby="title desc">
<title id="title">{title}</title><desc id="desc">WCM 内容审核工作台登录背景。左侧中部为文案留白，右侧为登录和注册表单留白；光学层、视频帧及识别结构构成抽象视觉。</desc>
<defs>{defs}</defs>{body}</svg>'''
    ET.fromstring(text)
    (OUT / name).write_text(text)


common = '''
<filter id="blur"><feGaussianBlur stdDeviation="55"/></filter>
<filter id="glow" x="-100%" y="-100%" width="300%" height="300%"><feGaussianBlur stdDeviation="8"/></filter>
<linearGradient id="edge" x1="0" y1="0" x2="1" y2="1"><stop stop-color="#c8edff" stop-opacity=".9"/><stop offset=".35" stop-color="#6589ff" stop-opacity=".12"/><stop offset=".7" stop-color="#6179ff" stop-opacity=".7"/><stop offset="1" stop-color="#a9d9ff" stop-opacity=".1"/></linearGradient>
'''

# A: a precision optical volume, cut from many fine elliptical sections.
defs = common + '''
<radialGradient id="base" cx=".37" cy=".72" r=".85"><stop stop-color="#132b60"/><stop offset=".5" stop-color="#0a142e"/><stop offset="1" stop-color="#050a17"/></radialGradient>
<linearGradient id="ring" x1="0" y1="0" x2="1" y2=".8"><stop stop-color="#6c91ff" stop-opacity=".03"/><stop offset=".23" stop-color="#4475ef" stop-opacity=".2"/><stop offset=".5" stop-color="#95dfff" stop-opacity=".95"/><stop offset=".62" stop-color="#7188ff" stop-opacity=".65"/><stop offset="1" stop-color="#344bb8" stop-opacity=".04"/></linearGradient>
<linearGradient id="pane" x1="0" y1="0" x2="1" y2="1"><stop stop-color="#3a5c9b" stop-opacity=".25"/><stop offset=".52" stop-color="#122348" stop-opacity=".6"/><stop offset="1" stop-color="#536abf" stop-opacity=".13"/></linearGradient>
'''
body = ['<path fill="url(#base)" d="M0 0h1920v1080H0z"/>', '<ellipse cx="590" cy="940" rx="590" ry="180" fill="#2859dc" opacity=".16" filter="url(#blur)"/>']
body.append('<g transform="translate(435 1050) rotate(-26)" fill="none" stroke="url(#ring)">')
for i in range(65):
    body.append(f'<ellipse rx="{620-i*3.8:.1f}" ry="{337-i*3.0:.1f}" cy="{-i*2:.1f}" stroke-width="{1+i*.019:.2f}" opacity="{.3+i*.01:.2f}"/>')
body.append('</g>')
for x,y,opacity in [(425,730,.38),(477,688,.56),(529,646,.95)]:
    body.append(f'<g transform="translate({x} {y}) skewY(-12)" opacity="{opacity}"><rect width="430" height="248" rx="22" fill="url(#pane)" stroke="url(#edge)"/><path d="M28 40h374M28 204h374" stroke="#96b8fa" opacity=".16"/>')
    for n in range(20):
        body.append(f'<rect x="{28+n*19}" y="17" width="8" height="6" rx="2" fill="#91b4ff" opacity=".26"/><rect x="{28+n*19}" y="222" width="8" height="6" rx="2" fill="#91b4ff" opacity=".26"/>')
    body.append('<path d="M162 68h-18v20m118-20h18v20M162 180h-18v-20m118 20h18v-20" fill="none" stroke="#b1ddff" stroke-width="2"/>')
    for n in range(34):
        t=n*pi*2/34
        px,py=212+39*cos(t),123+45*sin(t)
        body.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="1.8" fill="#c4eaff" opacity=".6"/>')
    body.append('<path d="M180 109l31 8 30-8M211 117v23l-9 2m-14 14q24 9 45 0" fill="none" stroke="#9ec9ff" opacity=".48"/><path d="M113 125h200" stroke="#c6f5ff" opacity=".65"/></g>')
body.append('<path d="M-120 294C355 80 755 12 1150-240M-90 320C355 120 755 45 1170-220" stroke="#658bd0" opacity=".13" fill="none"/><path d="M1080 0v1080" stroke="#bcd7ff" opacity=".04"/>')
svg('a-optical-depth.svg','A · 深蓝光学',defs,''.join(body))

# B: luminous silver space and layered glass video frames.
defs=common+'''
<linearGradient id="base" x1="0" y1="0" x2="1" y2="1"><stop stop-color="#f9fbff"/><stop offset=".5" stop-color="#eef2fa"/><stop offset="1" stop-color="#e3eafa"/></linearGradient>
<linearGradient id="glass" x1="0" y1="0" x2=".8" y2="1"><stop stop-color="#fff" stop-opacity=".98"/><stop offset=".48" stop-color="#dbe6fc" stop-opacity=".35"/><stop offset="1" stop-color="#b6c9f4" stop-opacity=".9"/></linearGradient>
<linearGradient id="blue" x1="0" y1="0" x2="1" y2="1"><stop stop-color="#d1e7ff"/><stop offset=".28" stop-color="#729eff"/><stop offset=".58" stop-color="#485beb"/><stop offset="1" stop-color="#10255e"/></linearGradient>
<linearGradient id="silver" x1="0" y1="0" x2="1" y2="1"><stop stop-color="#fff"/><stop offset=".44" stop-color="#bfcae0"/><stop offset=".65" stop-color="#fff"/><stop offset="1" stop-color="#7f9bd4"/></linearGradient>
<filter id="shadow" x="-50%" y="-50%" width="200%" height="220%"><feGaussianBlur stdDeviation="24"/></filter>
'''
body=['<path fill="url(#base)" d="M0 0h1920v1080H0z"/>','<ellipse cx="600" cy="1030" rx="460" ry="60" fill="#3865be" opacity=".15" filter="url(#shadow)"/>','<ellipse cx="1600" cy="100" rx="390" ry="130" fill="#fff" opacity=".75" filter="url(#blur)"/>']
body.append('<g transform="translate(665 842) rotate(-22)" fill="none">')
for i in range(27):
    body.append(f'<ellipse rx="{440-i*3.8:.1f}" ry="{201-i*2.7:.1f}" cy="{-i*.8:.1f}" stroke="url(#silver)" stroke-width="2" opacity=".6"/>')
body.append('</g>')
for x,y,scale,opacity in [(190,834,1,.4),(267,750,1,.68),(340,658,1,.98)]:
    body.append(f'<g transform="translate({x} {y}) matrix(1 .28 -.62 .67 0 0)" opacity="{opacity}"><rect x="0" y="9" width="516" height="290" rx="25" fill="#b2c4e8"/><rect width="516" height="290" rx="25" fill="url(#glass)" stroke="#fff" stroke-width="2"/><rect x="22" y="22" width="472" height="246" rx="16" fill="url(#blue)"/><path d="M45 60h425M45 231h425" stroke="#e9f3ff" opacity=".25"/>')
    for n in range(22):
        body.append(f'<rect x="{46+n*20}" y="37" width="8" height="6" rx="2" fill="#e1efff" opacity=".5"/>')
    body.append('<rect x="181" y="79" width="145" height="130" rx="14" fill="#173e96" opacity=".18"/><path d="M201 95h-9v15m105-15h14v15M201 194h-9v-15m105 15h14v-15" fill="none" stroke="#ebf6ff" stroke-width="2"/><ellipse cx="250" cy="137" rx="30" ry="38" fill="none" stroke="#f2f9ff" opacity=".65"/><path d="M224 125l26 8 27-8m-27 8v20m-15 7q15 7 30 0M202 149h100" fill="none" stroke="#eaf8ff" opacity=".65"/></g>')
body.append('<g fill="none" stroke="#9cb7e4" opacity=".18"><path d="M-50 127C310-50 620-39 910-100"/><path d="M-50 145C310-32 620-21 910-82"/><path d="M-50 163C310-14 620-3 910-64"/></g>')
svg('b-silver-frames.svg','B · 银白玻璃',defs,''.join(body))

# C: an original parametric face point cloud with a quiet scan plane.
defs=common+'''
<radialGradient id="base" cx=".35" cy=".8" r=".85"><stop stop-color="#123c40"/><stop offset=".5" stop-color="#091f26"/><stop offset="1" stop-color="#040f18"/></radialGradient>
<linearGradient id="scan" x1="0" y1="0" x2="0" y2="1"><stop stop-color="#83ffdc" stop-opacity="0"/><stop offset="1" stop-color="#7af4d7" stop-opacity=".13"/></linearGradient>
<linearGradient id="line" x1="0" y1="0" x2="1" y2="0"><stop stop-color="#8af7e1" stop-opacity="0"/><stop offset=".5" stop-color="#a9ffe9"/><stop offset="1" stop-color="#8af7e1" stop-opacity="0"/></linearGradient>
<pattern id="grid" width="64" height="64" patternUnits="userSpaceOnUse"><path d="M64 0H0v64" fill="none" stroke="#5aacb4" stroke-opacity=".045"/></pattern>
'''
body=['<path fill="url(#base)" d="M0 0h1920v1080H0z"/>','<path fill="url(#grid)" d="M0 0h1920v1080H0z"/>','<ellipse cx="694" cy="863" rx="200" ry="165" fill="#32b5b0" opacity=".075" filter="url(#blur)"/>']
for row in range(43):
    lat=-1.37+row*2.74/42
    for col in range(45):
        lon=-1.48+col*2.96/44
        xx=sin(lon)*cos(lat)
        yy=sin(lat)
        z=cos(lon)*cos(lat)
        nose=.43*exp(-((xx/.18)**2+((yy+.01)/.34)**2))
        sockets=.15*(exp(-(((xx-.34)/.2)**2+((yy+.2)/.13)**2))+exp(-(((xx+.34)/.2)**2+((yy+.2)/.13)**2)))
        z+=nose-sockets
        x=694+xx*170+z*70
        y=801+yy*198-z*11
        opacity=.14+.64*max(0,z/1.5)
        radius=.7+1.0*max(0,z/1.5)
        body.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{radius:.2f}" fill="#99f9e2" opacity="{opacity:.2f}"/>')
body.append('<path d="M539 648v-42h42m302 0h42v42M539 964v42h42m302 0h42v-42" stroke="#68c9bd" stroke-width="1.5" fill="none" opacity=".7"/><path d="M499 650h466v164H499z" fill="url(#scan)"/><path d="M467 814h530" stroke="url(#line)" stroke-width="2"/><path d="M467 814h530" stroke="url(#line)" stroke-width="9" filter="url(#glow)"/>')
for i in range(4):
    y=810+i*44
    body.append(f'<path d="M64 {y}h370" stroke="#65baa9" opacity=".12"/>')
    for j in range(35):
        h=5+19*(sin(j*.77+i)**2)
        body.append(f'<path d="M{65+j*10} {y-h:.1f}v{h*2:.1f}" stroke="#6ccebf" opacity="{.12+(j%6)*.025:.3f}" stroke-width="2"/>')
body.append('<path d="M90 742h320m-320-8v16m64-16v16m64-16v16m64-16v16m64-16v16m64-16v16" stroke="#6bbdb9" opacity=".3"/><path d="M0 164h890l132-132h898" fill="none" stroke="#70d9cb" opacity=".09"/>')
svg('c-signal-scan.svg','C · 青绿扫描',defs,''.join(body))

# D: a fluid, folded spectral ribbon with frame-like cross-sections.
defs=common+'''
<radialGradient id="base" cx=".38" cy=".76" r=".9"><stop stop-color="#292151"/><stop offset=".4" stop-color="#16152f"/><stop offset="1" stop-color="#0a0c1b"/></radialGradient>
<linearGradient id="ribbon" x1="0" y1=".8" x2="1" y2=".1"><stop stop-color="#3263ec" stop-opacity=".03"/><stop offset=".28" stop-color="#7185ff" stop-opacity=".4"/><stop offset=".5" stop-color="#ddc9ff" stop-opacity=".8"/><stop offset=".72" stop-color="#8270ec" stop-opacity=".42"/><stop offset="1" stop-color="#425ac8" stop-opacity=".03"/></linearGradient>
<linearGradient id="sheet" x1="0" y1="1" x2="1" y2="0"><stop stop-color="#3040c5" stop-opacity="0"/><stop offset=".48" stop-color="#a193ee" stop-opacity=".24"/><stop offset="1" stop-color="#6c79ff" stop-opacity=".06"/></linearGradient>
'''
body=['<path fill="url(#base)" d="M0 0h1920v1080H0z"/>','<ellipse cx="640" cy="955" rx="430" ry="150" fill="#7164ee" opacity=".15" filter="url(#blur)"/>','<path d="M-90 1120C230 460 728 1190 860 575S1010 66 1350-70L1610-50C1120 122 1190 396 1050 674S272 841-90 1240Z" fill="url(#sheet)"/>']
for i in range(72):
    t=i/71
    body.append(f'<path d="M{-130+t*140:.1f} {1090+t*210:.1f} C{175+t*70:.1f} {455+t*400:.1f} {742+t*280:.1f} {1210-t*190:.1f} {854+t*220:.1f} {576+t*125:.1f} S{997+t*250:.1f} {50+t*70:.1f} {1400+t*260:.1f} {-110+t*80:.1f}" fill="none" stroke="url(#ribbon)" stroke-width="{.8+sin(t*pi)*.8:.2f}" opacity="{.5+sin(t*pi)*.45:.2f}"/>')
body.append('<g transform="translate(596 771) rotate(-17) skewX(-12)" fill="none" stroke="url(#edge)">')
for i in range(5):
    body.append(f'<rect x="{i*14}" y="{-i*14}" width="240" height="140" rx="15" opacity="{.14+i*.09:.2f}"/>')
body.append('<path d="M133-13h-15v18m98-18h15v18M133 64h-15V46m98 18h15V46" stroke="#d8d7ff" opacity=".65"/></g>')
body.append('<path d="M-40 86C244 200 620 149 859-50" stroke="#bab3ff" opacity=".08" fill="none"/>')
svg('d-spectral-flow.svg','D · 靛紫流光',defs,''.join(body))
print('\n'.join(f'{p.name}: {p.stat().st_size:,} bytes' for p in OUT.glob('*.svg')))
