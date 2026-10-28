from pathlib import Path
from copy import deepcopy
from zipfile import ZipFile,ZIP_DEFLATED
import json,re,hashlib,csv,shutil
from lxml import etree
from docx import Document
from docx.shared import Pt,RGBColor
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
import pandas as pd
from latex2mathml.converter import convert

BASE=Path(r'E:\01-HUST\Nature\03\paper\水利学报\水利学报返修稿\返修工作文件')
for n in ['01_任务安排','02_论文修改','03_逐条回复','06_复现与核验']:(BASE/n).mkdir(exist_ok=True)
W='http://schemas.openxmlformats.org/wordprocessingml/2006/main'
M='http://schemas.openxmlformats.org/officeDocument/2006/math'
NS={'w':W,'m':M,'a':'http://schemas.openxmlformats.org/drawingml/2006/main','wp':'http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing','r':'http://schemas.openxmlformats.org/officeDocument/2006/relationships'}
XSL=etree.XSLT(etree.parse(r'C:\Program Files\Microsoft Office\root\Office16\MML2OMML.XSL'))
def el(n,**attrs):
    q=etree.Element('{'+W+'}'+n)
    for k,v in attrs.items():q.set('{'+W+'}'+k,str(v))
    return q
def r(text,red=True,bold=False):
    rr=el('r');pr=el('rPr');pr.append(el('rFonts',ascii='Times New Roman',hAnsi='Times New Roman',eastAsia='宋体'));pr.append(el('color',val='C00000' if red else '000000'))
    if bold:pr.append(el('b'))
    rr.append(pr);t=el('t');t.set('{http://www.w3.org/XML/1998/namespace}space','preserve');t.text=text;rr.append(t);return rr
def para(text,source=None,red=True,bold=False):
    p=el('p')
    if source is not None and source.find('w:pPr',NS) is not None:p.append(deepcopy(source.find('w:pPr',NS)))
    p.append(r(text,red,bold));return p
def mathp(tex,source=None):
    p=para('',source)
    p.remove(p[-1]);om=deepcopy(XSL(etree.fromstring(convert(tex).encode())).getroot())
    for rr in om.findall('.//m:r',NS):
        pr=rr.find('w:rPr',NS)
        if pr is None:pr=el('rPr');rr.insert(0,pr)
        pr.append(el('color',val='C00000'))
    p.append(om);return p
def textof(p):return ''.join(p.xpath('.//w:t/text() | .//m:t/text()',namespaces=NS))
SRC=BASE/'基于水能互补方程的河道洪水演算方法.docx'
with ZipFile(SRC) as z:parts={n:z.read(n) for n in z.namelist()}
root=etree.fromstring(parts['word/document.xml']);ps=root.findall('.//w:p',NS)
edits={
11:'从模型结构看，基于槽蓄关系的概念性方法通常将传播迟滞与洪水过程变形共同反映在河段整体响应中。随着演算距离增加，模型需要同时描述洪水过程的时间位移与形态变化。为此，可显式引入传播滞时，并以可供水量和潜在滞蓄水平共同约束水量的保留与释放。水能互补方程提供了一种满足双重约束及其极限条件的连续函数形式，其来源与Budyko框架下的水热平衡关系相联系[12-13]，相关形式已用于月径流、水文分割和降雨入渗等过程[14-16]。本文借鉴的是这一双重约束结构，并以河槽调蓄过程中的水量条件重新定义各变量。',
47:'式（8）沿用相关文献中的名称，称为水能互补方程[14-16]。在原水热平衡应用中，另一类约束可由潜在蒸散发等变量表征；推广至河道调蓄时，两类潜在量均以水量表示，分别对应当前可供水量与概念模型设定的潜在滞蓄水平。其中，潜在滞蓄量并非水流机械能，也不是由能量方程直接换算得到的河槽容量。因此，本文所称“水能互补”指具有文献来源的互补函数形式及其在蓄泄分配中的应用，其物理基础是水量平衡和滞蓄状态约束；该应用不构成由河道能量守恒方程导出的力学模型。',
49:'河道洪水演进过程可概化为传播滞时单元与有效调蓄单元的串联。上游来水经过有效传播滞时后进入调蓄单元，与前期有效滞蓄量共同构成当前可供水量；其中，一部分保留为时段末有效滞蓄量，其余部分形成出流。本文采用水能互补方程的双重约束函数形式描述这一水量分配过程，并结合水量平衡建立RWEC河道洪水演算模型。模型中的互补约束属于概念性本构假设，其合理性须通过边界条件、过程响应和独立资料检验。',
65:'初始有效滞蓄量根据演算起始阶段已知的入流、出流条件确定。在初始状态和模型参数给定后，联立式（9）—（12）可逐时段更新有效滞蓄量及出流量。这里调用的入流应理解为相应历史计算时段的平均入流，而非预报发布时刻的瞬时流量。在历史入流按计算要求及时获取、参数和初始状态已知的理想条件下，显式传播时滞使模型具有长度为τ的理论信息提前量。该提前量并非已经验证的业务有效预见期；观测误差、资料传输延迟、初始状态和参数误差均会影响实际预报性能。对于非整数时滞，还须保证构造入流平均量所用的全部资料在预报发布时刻已经可用，业务应用能力有待开展独立检验。',
233:'图5 各方案下三种模型在120 km断面的测试精度',
239:'以各组合测试期30场洪水洪峰滞时的中位数Tp为参照，比较C1、C2和C3在12个传播距离下的36组有效传播时间τ。复算得到Pearson相关系数为0.986，τ/Tp中位数为78.4%，与原分析一致。进一步计算τ−Tp及τ/Tp（图7），36组τ均小于Tp，τ−Tp的均值为−2.98 h，范围为−7.21～−0.44 h；τ/Tp的均值为77.4%，标准差为10.1个百分点，范围为41.4%～89.5%。C1、C2、C3的τ/Tp中位数分别为80.3%、70.4%和86.1%，表明这一比值随河槽形态而变，并非可以跨河槽直接采用的常数。沿程绝对差值总体增大，比例差异则不呈统一单调变化。高相关性反映两者对传播尺度变化的共同响应，不能据此将τ与完整洪峰滞时视为数值相等。',
241:'图7 有效传播时间与洪峰滞时的对应关系及其差异',
309:'图8和图9分别给出清江和笋溪河测试洪水的演算过程。清江两场洪水分别呈现双峰和单一主峰特征，RWEC能够跟随主要涨落过程。笋溪河两场测试洪水中，RWEC的全过程主峰流量相对误差分别为10.48%和1.08%，峰现时间误差分别为−1 h和0 h。第一场洪水以单一主峰为主，RWEC较好地给出峰现时间，但存在洪峰高估；第二场包含多个局部峰值，RWEC对全局主峰的幅值和时间拟合较好。两场结果尚不足以建立洪峰偏差与洪水类型之间的稳定统计关系。',
314:'k和n分别调节潜在滞蓄增量和互补曲线形态。对给定的前期状态与当前到达水量，增大k会提高潜在滞蓄量，并增加时段末有效滞蓄量，从而减少当期出流；这一局部作用不等同于整场洪水中始终具有更高滞蓄量，因为状态会随递推改变。对给定的两类约束，n增大时有效滞蓄量趋近其中较小者，因而n表征约束组合的曲线形态，而非某一项直接可测的水力参数。本文控制试验显示参数会随断面、比降、糙率和距离变化，但目前尚未建立由这些水力要素唯一确定k、n的关系。参数间还可能通过状态调整产生补偿，同样的过程拟合精度未必对应唯一的参数组合。跨河段迁移应同时检验河道条件相似性、流量范围、时间尺度与参数可辨识性，并以未参与率定的洪水检验；本文结果不能替代这一验证。',
316:'（1）本文采用水能互补方程的双重约束形式构建RWEC河道洪水演算模型，以有效传播时间表征显式时滞，以可供水量和潜在滞蓄量共同确定时段末有效滞蓄状态，并根据水量平衡计算出流。这里的互补关系是河道蓄泄分配的概念性本构表达，其应用不意味着直接引入或求解河道水流能量约束。',
317:'（2）HEC-RAS控制试验表明，在本文所采用的河槽形态、传播距离、河道比降及糙率条件下，RWEC保持较高演算精度。在120 km断面，RWEC、非线性马斯京根和线性马斯京根模型的平均NSE分别为0.97、0.80和0.74。36组有效传播时间与参照洪峰滞时高度相关，但全部小于参照值，τ/Tp随河槽形态和传播距离变化。显式传播时滞对应理想资料条件下的理论信息提前量，实际业务预见期仍须独立验证。现有三模型比较反映完整模型结构的差异，尚不能单独量化互补关系相对于显式时滞的独立增益。',
318:'（3）清江与笋溪河测试洪水中，RWEC的平均NSE分别为0.97和0.93，平均RMSE分别为577.57和9.43 m³/s。两河段结果支持模型在所检验样本中的过程模拟能力；笋溪河第一场测试洪水仍存在约10.48%的主峰高估。独立测试仅涵盖两个河段的四场洪水，尚不能据此确定对各类洪水的普遍适用性，也不能证明参数可以直接跨河段迁移。跨时间尺度稳定性、复杂水力条件下的表现及传播与调蓄作用的独立贡献，仍需更多独立资料和统一条件下的结构对照检验。'
}
edits[6]=textof(ps[6]).replace('基于水能互补方程构建RWEC河道洪水演算模型','借鉴水能互补方程的双重约束数学形式构建RWEC河道洪水演算模型')
edits[348]=textof(ps[348]).replace('based on the water-energy complementary equation, termed RWEC','using the dual-constraint mathematical form of the water-energy complementary equation, termed RWEC').replace('Effective storage is jointly constrained','The application is a conceptual storage relation rather than a derivation from the hydraulic energy equation. Effective storage is jointly constrained').replace('These results indicate that combining','The comparison evaluates the complete model structures and does not isolate the gain from the complementary relation alone. These results indicate that combining')
changed=[]
for idx,s in edits.items():
    old=ps[idx];new=para(s,old);old.getparent().replace(old,new);ps[idx]=new;changed.append({'原始段落':idx,'原文':textof(old),'修订':s})
def after(anchor,nodes):
    for n in nodes:anchor.addnext(n);anchor=n
after(ps[64],[para('k为无量纲系数，n为正的无量纲形状参数。由式（9）可知，有效滞蓄量不超过两类潜在量中的较小者；这一约束只作用于模型定义的有效滞蓄状态，不能直接等同于真实河段全部水体的蓄量。',ps[61])])
peak_eqs=[r'e_Q=\frac{Q_{\mathrm{p,sim}}-Q_{\mathrm{p,ref}}}{Q_{\mathrm{p,ref}}}\times100\%',r'e_t=t_{\mathrm{p,sim}}-t_{\mathrm{p,ref}}']
after(ps[82],[para('洪峰流量相对误差与峰现时间误差分别定义为：',ps[82]),*[mathp(x,ps[82]) for x in peak_eqs],para('正的流量误差表示高估，正的时间误差表示滞后；同时报告绝对误差。全过程主峰取相应序列的全局最大值，并列峰取最早时刻。对多峰洪水，另以参照过程的峰间谷点确定相邻峰的固定匹配窗口，在相同窗口内比较各模型的局部主峰，并将模拟主峰次序转换与同一峰的时间偏移区分。',ps[82])])
caption='注：箱体下、上边界分别为第25、第75百分位数，箱内横线为中位数，白色菱形为均值，旁列数值为均值；须线延伸至距箱体边界1.5倍四分位距以内的最远观测值，须线外散点为异常值。每个方案、模型包含30场测试洪水。'
after(ps[233],[para(caption,ps[233])])
dec_eqs=[r'\Delta T_{\mathrm{s}}=T_{\mathrm{p,sim}}-\tau',r'e_t=T_{\mathrm{p,sim}}-T_{\mathrm{p,ref}}=(\tau-T_{\mathrm{p,ref}})+\Delta T_{\mathrm{s}}']
after(ps[239],[para('为区分模型串联结构中的峰时调整，将纯时间平移后的峰现时刻作为传播单元输出，再计算调蓄单元引起的峰时变化：',ps[239]),*[mathp(x,ps[239]) for x in dec_eqs],para('这里的两类峰时调整构成代数分解，不是河道平流与扩散的唯一物理归因。对C1、C2、C3各360个测试过程，调蓄峰时调整量的均值分别为4.96、4.48和4.02 h，纯平移相对于参照洪峰滞时的平均差分别为−3.31、−3.37和−2.36 h，两项叠加后最终平均峰时误差分别为1.65、1.11和1.65 h。因此，调蓄单元总体补偿了纯平移所留下的提前量，但部分补偿超过参照峰时，形成正的最终时间误差。逐场分解满足上述恒等式。',ps[239])])
sunparas=[para('对第二场笋溪河洪水，两种马斯京根模型的全过程最大模拟峰出现在后一个局部峰附近，直接比较全局主峰会得到30 h的时间差；以参照主峰相邻谷点限定相同匹配窗口后，两种模型在该窗口的峰时误差均为0 h，峰值误差分别为−34.25%和−29.03%。这一现象反映主次峰幅值排序的变化，不能仅解释为同一洪峰整体滞后30 h。',ps[309])]
after(ps[309],sunparas)
# Normalize text units and range signs without reconstructing existing math.
unitpat=re.compile(r'(?<=\d)(?=(?:km|m³/s|m3/s|h)(?![A-Za-z]))')
for t in root.findall('.//w:t',NS):
    if not t.text:continue
    old=t.text;s=unitpat.sub(' ',old)
    s=re.sub(r'(?<=\d)[–](?=\d)','～',s)
    if s!=old:
        t.text=s;rr=t.getparent();pr=rr.find('w:rPr',NS)
        if pr is None:pr=el('rPr');rr.insert(0,pr)
        oldc=pr.find('w:color',NS)
        if oldc is not None:pr.remove(oldc)
        pr.append(el('color',val='C00000'))
# Units split across differently formatted runs still need one visible space.
for p in root.findall('.//w:p',NS):
    ts=p.findall('.//w:t',NS);joined=''.join(t.text or '' for t in ts)
    positions=[m.start() for m in unitpat.finditer(joined)]
    offset=0
    for t in ts:
        val=t.text or '';local=[x-offset for x in positions if offset<=x<offset+len(val)]
        for j in reversed(local):val=val[:j]+' '+val[j:]
        if local:
            t.text=val;pr=t.getparent().find('w:rPr',NS)
            if pr is None:pr=el('rPr');t.getparent().insert(0,pr)
            cr=pr.find('w:color',NS)
            if cr is None:pr.append(el('color',val='C00000'))
            else:cr.set('{'+W+'}val','C00000')
        offset+=len(t.text or '')-len(local)
# Keep table captions with their tables and Figure 5's three panels together.
def keepnext(p):
    pr=p.find('w:pPr',NS)
    if pr is None:pr=el('pPr');p.insert(0,pr)
    if pr.find('w:keepNext',NS) is None:pr.append(el('keepNext'))
for p in root.findall('.//w:p',NS):
    if re.match(r'^表[1-7] ',textof(p)):keepnext(p)
    for b in p.findall('.//a:blip',NS):
        # The three raster panels form one figure; avoid splitting them over pages.
        rid=b.get('{'+NS['r']+'}embed')
        source_rels=etree.fromstring(parts['word/_rels/document.xml.rels'])
        targets={x.get('Id'):x.get('Target') for x in source_rels}
        if targets.get(rid,'').endswith(('image5.jpeg','image6.jpeg','image7.jpeg')):keepnext(p)
keepnext(ps[233])
# Replace Figure 4 and 7 media in their existing relationships.
rels=etree.fromstring(parts['word/_rels/document.xml.rels'])
relmap={x.get('Id'):x.get('Target') for x in rels}
figuremap={'word/media/image4.png':BASE/'05_修订图件/图4_120km刻度修订.png','word/media/image10.png':BASE/'05_修订图件/图7_残差与比值补充.png'}
for part,f in figuremap.items():parts[part]=f.read_bytes()
# Figure 6 originally occupies two separate paragraphs. Consolidate its six panels.
blips=root.findall('.//a:blip',NS);p6=[]
for b in blips:
    target=relmap.get(b.get('{'+NS['r']+'}embed'),'')
    if target.endswith(('image8.jpeg','image9.jpeg')):
        pp=b
        while pp is not None and pp.tag!='{'+W+'}p':pp=pp.getparent()
        p6.append((target,pp))
from PIL import Image
png=Image.open(BASE/'05_修订图件/图6_120km刻度修订.png').convert('RGB')
import io
buf=io.BytesIO();png.save(buf,format='JPEG',quality=97)
parts['word/media/image8.jpeg']=buf.getvalue()
for target,pp in p6:
    if target.endswith('image9.jpeg'):
        for child in list(pp):
            if child.tag!='{'+W+'}pPr':pp.remove(child)
    else:
        for crop in pp.findall('.//a:srcRect',NS):crop.getparent().remove(crop)
        for ext in pp.findall('.//wp:extent',NS)+pp.findall('.//a:xfrm/a:ext',NS):
            cx=int(ext.get('cx'));ext.set('cy',str(round(cx*4.7/7.2)))
# Figure 7 changed from square to three panels.
for b in root.findall('.//a:blip',NS):
    target=relmap.get(b.get('{'+NS['r']+'}embed'),'')
    if target.endswith('image10.png'):
        inline=b
        while inline is not None and inline.tag not in ['{'+NS['wp']+'}inline','{'+NS['wp']+'}anchor']:inline=inline.getparent()
        if inline is not None:
            for ext in inline.findall('.//wp:extent',NS)+inline.findall('.//a:xfrm/a:ext',NS):ext.set('cx','5943600');ext.set('cy',str(round(5943600*2.6/7.2)))
parts['word/document.xml']=etree.tostring(root,xml_declaration=True,encoding='UTF-8',standalone=True)
# Attach working comments to unresolved numerical issues in the manuscript.
comments=etree.Element('{'+W+'}comments',nsmap={'w':W})
comment_items=[(276,'数值来源待核','送审稿清江RWEC平均RMSE为577.57 m³/s，而目前找到的过程数据复算为532.57 m³/s。现值暂按送审稿保留，尚不能作为已复现结果。请按任务A锁定最终送审过程、参数及时间单位。'),(100,'表1率定边界待核','本表n范围为0～5，但图4源数据中n最大7.73。HEC及笋溪河存档的实际搜索边界均与此表不同。须以本次各试验真实代码和日志改表，不得修改已率定数值来迁就边界。'),(65,'平均入流实现待统一','本段已按理论平均量改写。笋溪河现行代码实际使用平移后插值点值，需先核定输入记录是否为时段平均值，再统一数值实现与预见期资料可用性。仅文字修改尚不能关闭R2-10。')]
for cid,(idx,title,body) in enumerate(comment_items):
    p=ps[idx];start=el('commentRangeStart',id=cid);end=el('commentRangeEnd',id=cid)
    pos=1 if len(p) and p[0].tag=='{'+W+'}pPr' else 0;p.insert(pos,start);p.append(end)
    ref=el('r');ref.append(el('commentReference',id=cid));p.append(ref)
    c=el('comment',id=cid,author='返修核查',date='2026-10-28T00:00:00Z');c.append(para(title+'：'+body,red=False));comments.append(c)
parts['word/comments.xml']=etree.tostring(comments,xml_declaration=True,encoding='UTF-8',standalone=True)
rel=etree.Element('{http://schemas.openxmlformats.org/package/2006/relationships}Relationship',Id='rIdRevisionComments',Type='http://schemas.openxmlformats.org/officeDocument/2006/relationships/comments',Target='comments.xml');rels.append(rel)
parts['word/_rels/document.xml.rels']=etree.tostring(rels,xml_declaration=True,encoding='UTF-8',standalone=True)
ct=etree.fromstring(parts['[Content_Types].xml']);ct.append(etree.Element('{http://schemas.openxmlformats.org/package/2006/content-types}Override',PartName='/word/comments.xml',ContentType='application/vnd.openxmlformats-officedocument.wordprocessingml.comments+xml'))
parts['[Content_Types].xml']=etree.tostring(ct,xml_declaration=True,encoding='UTF-8',standalone=True)
parts['word/document.xml']=etree.tostring(root,xml_declaration=True,encoding='UTF-8',standalone=True)
MAN=BASE/'02_论文修改/基于水能互补方程的河道洪水演算方法_返修工作稿_红字_20261028.docx'
with ZipFile(MAN,'w',ZIP_DEFLATED) as z:
    for n,b in parts.items():z.writestr(n,b)

# Responses retain the source template's structural order and paragraphs.
TEMPLATE=BASE/'水利学报_jhe20260716_审稿意见逐条回复模板.docx'
with ZipFile(TEMPLATE) as z:rparts={n:z.read(n) for n in z.namelist()}
rr=etree.fromstring(rparts['word/document.xml']);rp=rr.findall('.//w:p',NS)
response={
6:'工作版说明：本文件逐条记录已完成修改与仍需补算或核实的内容，尚不能直接作为投稿回复。已完成内容对应本轮红字工作稿；其余事项保持待核状态。当前稿件题目按作者意见保留。',
8:'感谢编辑及两位审稿专家的审阅。我们按照13条意见逐项修订，并补充了36组传播时间差异分析、1080个测试过程的峰时分解及笋溪河逐场洪峰评价。下文说明实际完成的修改，同时列明尚需核实的数值来源和补充对照试验。本稿是返修工作版，正式提交前将删除全部工作状态说明并统一页码。',
14:'作者回复：感谢专家指出术语对应关系不清的问题。我们保留原题，同时在引言、第2章末段和第3章开头明确：本文借鉴的是已有水能互补方程的双重约束数学形式。可供水量表示水量供给约束，潜在滞蓄量表示概念性滞蓄水平；后者不是水流机械能，也不是直接从能量方程得到的河槽容量。修订文字将数学形式推广与力学推导区分，并说明本构假设需由边界条件、过程响应和独立资料检验。',
15:'修改内容及位置：已修改；引言第二段、第2章式（8）后、第3章开头、中文摘要、英文摘要及结论（1）。论文题目保留。',
18:'作者回复：我们同意，原有对比不能独立识别显式传播时滞与互补关系的贡献。这一条尚未完成补算，当前不宣称互补关系具有已被独立证实的精度增益。已制定两组对照：固定与RWEC相同的时滞后分别率定Lag-MK和Lag-NMK的其余参数；再允许两种新增模型独立率定时滞，检查固定时滞是否造成结构偏向。各组必须采用相同资料、初始条件、时间离散、率定目标和测试划分，并记录参数数目、数值可行性和边界命中。补算后依据实际结果修订摘要、结果及结论。',
19:'修改内容及位置：待补算；任务A。第4.1节及第4.2、4.3节待加入对照结果。现阶段已在结论（2）限定原有三模型比较的归因范围。',
22:'作者回复：已补充带符号的洪峰相对误差和峰现时间误差定义，并复算笋溪河两场测试洪水的三模型指标。RWEC两场洪峰误差为10.48%和1.08%，峰时误差为−1 h和0 h。多峰场次同时报告全局主峰与同一参照峰窗口的指标：两种马斯京根模型全局最大峰发生主次峰转换，得到30 h的全局时间差；在相同参照主峰窗口内，两者峰时误差均为0 h。这两类指标含义不同，已予区分。清江逐时结果尚未匹配送审稿RWEC平均RMSE，故其新增数字暂不写入稿件；四场洪水全套指标完成后再关闭本条。',
23:'修改内容及位置：部分完成；第4.1节评价方法、第4.3节图8和图9前的分析。已输出笋溪河逐场指标和多峰匹配窗口；清江指标待任务A核定资料后补齐。',
28:'作者回复：已按模型串联结构逐场分解1080个HEC-RAS测试过程。调蓄峰时调整量定义为模拟峰滞时减去τ，最终误差等于纯平移相对于参照峰滞时的差与该调整量之和。C1、C2、C3的调蓄调整均值分别为4.96、4.48和4.02 h；纯平移差的均值分别为−3.31、−3.37和−2.36 h；最终平均峰时误差分别为1.65、1.11和1.65 h。逐场代数核验通过。该分解说明调蓄单元对峰时的补偿及过度补偿，不是对实际平流与扩散作用的唯一归因。天然河段的完整分解尚待清江资料核定。',
29:'修改内容及位置：HEC部分已完成；第4.2节（4）新增峰时分解公式及定量分析。天然河段部分仍待补齐。',
32:'作者回复：已将适用性结论限定于所检验样本。笋溪河第一场以单一主峰为主，RWEC有10.48%的峰值高估且提前1 h；第二场含多个局部峰，全局主峰误差为1.08%，峰时误差为0 h。现有样本不足以建立误差与洪水类型的稳定统计关系，故不以这两场结果推断特定类型普遍适用或不适用。清江对应逐场指标待资料来源核定后补充；最终将结合四场过程特征描述偏差，并保留样本量限制。',
33:'修改内容及位置：部分完成；第4.3节图8和图9前的分析及结论（3）。清江过程特征与新增指标待核。',
36:'作者回复：已在第3章和第4.3节补充k、n的数学作用及迁移限制。在其他即时条件固定时，k增大使潜在滞蓄量和有效滞蓄量增加，但递推状态变化后不能简单断言整场蓄量始终更高；对给定两类约束，n增大使有效滞蓄量趋近较小约束。k、n是有物理指向的概念参数，尚不能由比降、糙率及断面形态唯一确定。现有控制试验给出响应趋势，尚未建立区域移植公式。补充文字明确了参数补偿、可辨识性及独立资料验证的必要性。',
37:'修改内容及位置：已修改；第3章式（11）后的解释、第4.3节末段和结论（3）。本轮未宣称已建立定量参数迁移关系。',
40:'作者回复：已重绘图4及图6，各子图均明确显示120 km刻度，且120 km数据点完整可见。重绘使用原有图源数据，未改动率定节点，也未补入插值节点。',
41:'修改内容及位置：已修改；图4、图6。修订图件同时保留可编辑SVG、PDF及600 dpi TIFF。',
44:'作者回复：已将第3章的预见期解释限定为参数和初始状态已知、历史入流按计算要求及时获得时的理论信息提前量，并明确不等同于已经验证的业务有效预见期。新增文字说明观测误差、传输延迟、初始状态与参数误差对预报的影响，并要求非整数时滞的构造只能使用发布时刻已可用的资料。本文未新增业务预报实测检验，相关结论保持限定。',
45:'修改内容及位置：已修改；第3章式（12）后及结论（2）。资料插值方式的数值一致性另列于审稿人2意见10，仍需核查。',
48:'作者回复：已检查并统一正文、摘要、表题图题中数值与km、h、m³/s等单位的间距，修订文字使用半角空格。新绘图件英文和数字采用Times New Roman，中文采用宋体。',
49:'修改内容及位置：已修改并抽查；摘要、第4.2和4.3节、表3、图5图题、结论等。最终合稿还须在页码锁定后作一次全文排版复核。',
52:'作者回复：已统一表1和表3数值范围连接符为“～”，并检查同类数值范围的正文写法。负数符号及文献页码连接符保持其本来含义。',
53:'修改内容及位置：已修改；表1、表3及相应数值区间。另发现表1的实际搜索边界需重新核对，已作为数值复核事项单独登记。',
56:'作者回复：已从原始HEC-RAS逐时过程复算每个组合30场测试洪水的峰滞时中位数，形成36组对照，并扩展图7。Pearson相关系数复算为0.985673，τ/Tp中位数为78.37%。36组τ−Tp均为负，均值−2.98 h，范围−7.21～−0.44 h；τ/Tp均值77.45%，样本标准差10.08个百分点。C1、C2、C3的比值中位数为80.27%、70.39%和86.14%，范围分别为58.26%～84.65%、41.44%～77.55%和76.95%～89.46%。因此比值存在河槽间差异，不能以高相关性支持数值一致或使用统一比例。图7补充了按距离排列的差值和比值，分组统计表及逐组数据已归档。',
57:'修改内容及位置：已完成；第4.2节（4）和扩展后的图7。另附36组原始分析表及按河槽、距离的统计表。',
60:'作者回复：已核对图源程序和原图中的统计标记，在图5图注中补充箱体第25/75百分位数、中位线、白色均值菱形、旁列均值数字、1.5倍四分位距须线及异常散点的含义，并注明每组包含30场测试洪水。',
61:'修改内容及位置：已修改；图5图注。',
64:'作者回复：已删除第3章“获取实时入流”的表述，明确理论公式中的变量为相应历史计算时段的平均入流，并补充发布时刻资料可用性的要求。代码核查发现，HEC-RAS通用实现与笋溪河复现实现对入流离散方式的处理并不相同：后者实际调用平移后的插值点值。因此，仅修改措辞尚不能关闭本条。下一步须核定原始记录为时段平均量还是瞬时值，统一入流构造并复算受影响结果，不能把点值实现直接写成已采用时段积分或平均。',
65:'修改内容及位置：文字已修改，数值一致性待核；第3章式（10）—（12）及其后说明。任务A将核定数据时间含义，并执行非整数时滞与资料可用性检查。',
66:'本工作版已据实际完成内容更新回复，未将计划中的对照试验写成既成结果。待资料与数值一致性核查、新增对照及四场洪峰统计完成后，将统一修订摘要、图表、结论及具体页码，再形成正式提交回复。'
}
for i,s in response.items():
    old=rp[i];new=para(s,old,red=(i in [6,19,23,29,33,65]));old.getparent().replace(old,new)
# The optional submission checklist is maintained in the task files instead.
for p in list(rr.findall('.//w:p',NS)):
    txt=textof(p)
    if txt.startswith(('本工作版已据实际完成内容更新回复','附：提交前核对清单','□')) or txt=='全体作者':
        p.getparent().remove(p)
for p in rr.findall('.//w:p',NS):
    t=textof(p)
    if re.match(r'^意见\s*\d+$',t) or t.startswith('审稿意见：'):keepnext(p)
# Use black headings as required; preserve underlying template package parts.
for p in rr.findall('.//w:p',NS):
    style=p.find('w:pPr/w:pStyle',NS)
    if style is not None and style.get('{'+W+'}val') in ['Title','Heading1','Heading2']:
        for cr in p.findall('.//w:color',NS):cr.set('{'+W+'}val','000000')
rparts['word/document.xml']=etree.tostring(rr,xml_declaration=True,encoding='UTF-8',standalone=True)
RESP=BASE/'03_逐条回复/水利学报_jhe20260716_逐条回复_工作版_20261028.docx'
with ZipFile(RESP,'w',ZIP_DEFLATED) as z:
    for n,b in rparts.items():z.writestr(n,b)

alltex=r'''% 本轮新增公式 原稿现有公式在Word中保留可编辑OMML
% 洪峰流量相对误差
e_Q=\frac{Q_{\mathrm{p,sim}}-Q_{\mathrm{p,ref}}}{Q_{\mathrm{p,ref}}}\times100\%
% 峰现时间误差
e_t=t_{\mathrm{p,sim}}-t_{\mathrm{p,ref}}
% 模型结构峰时分解
\Delta T_{\mathrm{s}}=T_{\mathrm{p,sim}}-\tau
e_t=T_{\mathrm{p,sim}}-T_{\mathrm{p,ref}}=(\tau-T_{\mathrm{p,ref}})+\Delta T_{\mathrm{s}}
% 时段平均量的理论定义 以下为待统一数值实现的定义 不声称现有代码均已实现
\bar I_{t-\tau}=\frac{1}{\Delta t}\int_{t-\tau-\Delta t}^{t-\tau}I(s)\,\mathrm{d}s
% 对原互补式的等价表达 供核对参数解释
S_t=(S_{W,t}^{-n}+S_{E,t}^{-n})^{-1/n}
'''
(BASE/'02_论文修改/新增公式_LaTeX可编辑源.tex').write_text(alltex,encoding='utf-8')
original_tex=r'''\documentclass{article}
\usepackage{amsmath,amssymb}
\begin{document}
% Source equations transcribed for editable reuse. The original Word OMML equations remain authoritative.
\begin{equation}\frac{X-X_0}{X_P-X_0}=\frac{Y}{Z-X_0}\tag{1}\end{equation}
\begin{equation}\frac{E}{E_P}=\frac{R}{P}\tag{2}\end{equation}
\begin{equation}\frac{E}{P}+\frac{E}{E_P}=1\tag{3}\end{equation}
\begin{equation}\frac{E}{E_P}=\frac{P}{(P^n+E_P^n)^{1/n}}\tag{4}\end{equation}
\begin{equation}(E/P)^n+(E/E_P)^n=1\tag{5}\end{equation}
\begin{equation}H/H_W\to1\ (\eta\to0),\quad H/H_E\to1\ (\eta\to\infty)\tag{6}\end{equation}
\begin{equation}\begin{aligned}\partial H/\partial H_W&\to1,\quad\partial H/\partial H_E\to0\quad(\eta\to0),\\\partial H/\partial H_W&\to0,\quad\partial H/\partial H_E\to1\quad(\eta\to\infty)\end{aligned}\tag{7}\end{equation}
\begin{equation}(H/H_W)^n+(H/H_E)^n=1\tag{8}\end{equation}
\begin{equation}(S_t/S_{W,t})^n+(S_t/S_{E,t})^n=1\tag{9}\end{equation}
\begin{equation}S_{W,t}=S_{t-1}+\bar I_{t-\tau}\Delta t\tag{10}\end{equation}
\begin{equation}S_{E,t}=S_{t-1}+k\bar I_{t-\tau}\Delta t\tag{11}\end{equation}
\begin{equation}\bar Q_t=(S_{W,t}-S_t)/\Delta t\tag{12}\end{equation}
\begin{equation}W_t=K[X\bar I_t+(1-X)\bar Q_t]\tag{13}\end{equation}
\begin{equation}W_t=K[X\bar I_t+(1-X)\bar Q_t]^m\tag{14}\end{equation}
\begin{equation}W_t-W_{t-1}=(\bar I_t-\bar Q_t)\Delta t\tag{15}\end{equation}
\begin{equation}x=(t-t_s)/D,\quad0\le x\le1\tag{16}\end{equation}
\begin{equation}f(x;\alpha,\beta)=x^{\alpha-1}(1-x)^{\beta-1}/B(\alpha,\beta)\tag{17}\end{equation}
\begin{equation}g(x)=f(x;\alpha,\beta)/\max_{0\le x\le1}f(x;\alpha,\beta)\tag{18}\end{equation}
\begin{equation}I(t)=Q_b+(Q_p-Q_b)g(x)\tag{19}\end{equation}
\begin{equation}\alpha=1+r(c-2),\quad\beta=1+(1-r)(c-2)\tag{20}\end{equation}
\end{document}
'''
(BASE/'02_论文修改/原稿20式_LaTeX可编辑源.tex').write_text(original_tex,encoding='utf-8')
pd.DataFrame(changed).to_csv(BASE/'06_复现与核验/正文修改对照.csv',index=False,encoding='utf-8-sig')

# Original review notice, with exact wording retained separately.
notice=Path('work/review_notice.txt').read_text(encoding='utf-8')
(BASE/'01_任务安排/编辑部原始通知与13条意见.txt').write_text(notice,encoding='utf-8')
tasks=[
('R1-1','术语与题目','已修改','C','保留原题 说明数学形式与能量机理的区别','引言 第2和3章 摘要 结论'),
('R1-2','相同时滞对照','待补算','A','完成Lag-MK Lag-NMK固定时滞及独立时滞试验','第4.1至4.3节'),
('R1-3','逐场洪峰指标','部分完成','B','笋溪河完成 清江待匹配送审结果 全局峰和窗口峰同时保留','第4.1和4.3节'),
('R2-1','峰时贡献分解','部分完成','B','HEC1080过程完成 天然河段待清江基准','第4.2节4'),
('R2-2','洪峰偏差与适用范围','部分完成','B','已收敛结论 清江统计待补 不做小样本显著性推断','第4.3节 结论3'),
('R2-3','参数物理意义与移植','已修改','C','说明固定状态的局部数学作用 参数补偿及移植前提','第3章 第4.3节末'),
('R2-4','120km横轴','已修改','D','图4和图6全部子图显示120 km 保留原节点','图4 图6'),
('R2-5','理论预见期','已修改','C','限定理想信息提前量 补充误差和延迟影响','第3章 结论2'),
('R2-6','量与单位空格','已修改','D','正文表图题已统一 投稿前再核图内单位','全文'),
('R2-7','范围号','已修改','D','表1表3统一为～ 不改变负号与文献页码','表1 表3'),
('R2-8','tau/Tp离散与偏差','已完成','B','36组 1080过程复算 分组统计 扩展图7','第4.2节4 图7'),
('R2-9','箱线图说明','已修改','D','1.5IQR 白菱形均值 均值数字 异常散点 30样本','图5图注'),
('R2-10','平均入流与实现','部分完成','A','文字已改 笋溪河插值点值与理论平均量须统一','第3章式10至12')]
pd.DataFrame(tasks,columns=['意见编号','修改对象','状态','阶段','完成标准及剩余工作','正文位置']).to_csv(BASE/'01_任务安排/13条审稿意见任务台账.csv',index=False,encoding='utf-8-sig')

plan='''# 水利学报返修任务与当前进度

稿件 jhe20260716。编辑部通知提交期限为2026年11月10日；本轮工作日期为2026年10月28日。按作者选择保留原题。本目录为持续返修的主工作目录，原始两份Word保留不变。

当前已完成8条意见的文字、图件或分析修改；另5条仍有补算或资料核实事项。红字工作稿和逐条回复均为中间版本，不能直接提交。回复中的“已完成”只用于确实完成的事项。

## 当前可审阅文件

- 02_论文修改：红字工作稿、新增公式的LaTeX源。
- 03_逐条回复：按原顺序逐一说明13条意见的处理结果及剩余事项。
- 04_补充分析：36组τ/Tp、1080个测试过程的峰时分解、笋溪河逐场洪峰误差、来源哈希。
- 05_修订图件：图4、图6、扩展图7的可编辑SVG/PDF及TIFF。
- 06_复现与核验：修改对照、计算和绘图脚本、核验记录。

## 先处理的三个数值一致性问题

1. 清江送审稿写RWEC平均RMSE 577.57 m³/s、k=0.02、n=1.13、τ=4.17 h。已定位的绘图源和若干存档参数不一致，其中一套绘图源复算平均RMSE为532.57 m³/s。此值仅存为候选核查结果，尚未覆盖原稿577.57。应先从最终送审版本追溯到逐时模拟与完整参数，不得任取较优结果替换。
2. 理论式（10）（11）使用时段平均入流，而笋溪河复现代码调用平移后的插值点值。先核定输入记录的时间含义；若是时段平均流量，可按平均量序列构造时间移位；若是瞬时采样，须用一致的积分或离散平均。不能只改文字就宣称数值实现一致。非整数时滞不能使用发布时刻之后才能取得的观测值。
3. 表1写k、n均为0～5，但图4的数据n最大为7.73，图6部分值更高。HEC存档边界为k=10^-6～100、n=0.01～300、τ=0～150 h；笋溪河现行版本又有不同边界。应逐试验核定实际代码和日志，并按事实修订表1，不能为了让表格一致而修改已率定参数。

## 阶段安排

阶段A 优先解决基准和离散一致性，再补显式时滞对照。建议10月28日至11月1日完成。输出统一数据和参数清单、基线复算表、固定时滞与独立时滞的逐场结果。详见任务A。

阶段B 完成全部峰值指标、峰时分解和洪水特征描述。建议11月1日至3日完成。现有HEC与笋溪河分析可复用；清江需等待A的基准锁定。多峰匹配须由参照过程事先确定，不随模型改变。详见任务B。

阶段C 合并论文与13条回复。建议11月3日至6日完成。正文以自然连贯的论文语言表述，不将工作安排写入论文。新对照的结果必须同步影响中英文摘要、结果与结论，不能预设RWEC占优。详见任务C。

阶段D 投稿前核验。建议11月6日至9日完成。统一公式编号、图表、正文、回复位置和最终页码，逐页检查，并留出11月10日提交缓冲。详见任务D。以上日期是建议安排，并未创建定时任务或自动提交。

## 继续工作的顺序

先读本文件与13条审稿意见任务台账，再读04_补充分析的核查摘要和任务A。现有旧目录只读取资料，新增计算及返修文件均写入本目录。完成每阶段后更新台账与逐条回复，保留失败记录，不把未完成试验写成已完成。
'''
(BASE/'00_返修总控与继续入口.md').write_text(plan,encoding='utf-8')
taska=r'''# 任务A 核定数值基准并补显式时滞对照

对应R1-2、R2-10，并为R1-3与R2-1提供基准。先解决总控文件所列三处不一致，再启动全量率定，避免建立在不同输入和离散约定之上的表面对比。

固定原70/30合成洪水划分、清江6/2和笋溪河7/2划分。笋溪河当前依据“当前笋溪河数据基准.md”为20260920的F08配平直跑版。输入数据只读，保留原始数值与哈希。不另挑测试事件，不因对照结果调整事件边界。

基线复算需要逐事件重现原NSE、KGE、RMSE，参数、时间单位和初始状态必须来自同一实现。明确原始流量是瞬时值还是时段平均值，τ非整数时怎样构造历史时段入流。用独立的时间戳可用性核查证明所有插值支点在预报发布时刻已知；若只能采用此前已知值，应在独立业务试验中验证，不将后验插值结果当实时预报效果。

第一组机制对照令Lag-MK、Lag-NMK采用与RWEC同一显式τ，另率定各自蓄泄参数。第二组允许新增两模型独立率定τ，检验固定RWEC时滞带来的偏向。保留原MK、NMK结果，报告五模型表现及新增两模型的两种率定设置。

Lag-MK采用W=K[X I_lag+(1−X)Q]，Lag-NMK采用W=K[X I_lag+(1−X)Q]^m，并联立一致的离散水量平衡。新实现不能仅平移已有出流曲线来代替重新率定。所有模型使用同一时间网格、平均量定义、初始出流资料和同一可行性标准；为减少传统MK稳定性对步长的限制，可在所有模型统一的细时间网格演算后以相同规则聚合评价，并另列原稿配置作为敏感性比较。若采用子步长，其结果和额外数值设置必须公开。

非线性模型不得用裁剪负流量代替有效求根，报告无效参数比例和搜索边界。K在m不等于1时的量纲不是单纯时间，按实际存储和流量单位注明，或以带固定参考流量的Tref参数化并给出换算。不能新增率定参数后仍声称与三参数RWEC参数数目完全相同。

率定目标为率定场次等权平均NSE，测试参数冻结。至少保留3个固定随机种子检验寻优稳定性，记录预算及未收敛标志；不把多种子当独立洪水样本。输出逐场指标、过程、参数及边界诊断，配对报告模型指标差，按断面、河槽、河段统计。不预设模型优劣，不对相关断面作独立重复的显著性检验。

完成标准：基线能重现，入流离散和量纲一致，两套新增对照结果齐全，所有失败有记录，正文和回复中的互补独立增益表述与实际结果相符。
'''
taskb='''# 任务B 补齐洪峰统计和机理分析

对应R1-3、R2-1、R2-2、R2-8。36组τ/Tp及HEC1080个测试过程的结构峰时分解已完成；结果在04_补充分析，来源均有哈希。

笋溪河逐场指标已经核算。全局主峰取各序列全局最大值，并列峰取最早时刻。多峰匹配窗口只由参照下游过程确定：以幅度突出程度不低于参照流量极差10%、间距不小于3条记录识别局部峰，用相邻峰间谷点划定主峰窗口，对所有模型使用同一窗口。此阈值是本轮诊断规则，需作5%、10%、15%的敏感性检查并人工核查，不能称为普适标准。全局结果和匹配结果同时保留，说明主次峰次序变化。

清江在任务A锁定送审基准后补入同样统计。记录洪峰流量、峰现时间、涨水历时、最大涨率、峰数和过程展宽；样本仅四场，采用逐场描述，不建立小样本泛化分类或显著性关系。

结构分解为DeltaTs=Tp_sim−tau，et=(tau−Tp_ref)+DeltaTs。对每场核验恒等式，报告正负分量及抵消，不将调蓄调节称为唯一扩散归因。多峰情况下对应同一参考峰，不把跨峰切换作为同一波的传播时间。

按36个河槽与距离组合报告tau/Tp和tau−Tp；同一河槽不同距离共享洪水输入，不能将其当作36个完全独立河流样本计算泛化置信区间。区分与1∶1线的差、回归残差及最终峰时误差。现有图7采用前两者中的“与1∶1线的差”，不是回归残差。

完成标准：清江和笋溪河四场三模型及新增对照的指标齐全，峰匹配规则明确，正文只作样本可支持的解释，所有数值可从逐时序列复算。
'''
taskc='''# 任务C 合并论文与逐条回复

对应R1-1、R2-3、R2-5和全部意见的回复。本轮按作者要求保留原题，已补充互补数学形式和河道能量方程的区别。不要再次擅自改题。

以02_论文修改的红字稿为基础，逐条对照台账。将任务A、B的新结果同步到摘要、试验方法、图表、讨论、结论及英文摘要。若相同时滞对照不支持互补独立增益，须收敛创新定位，不回避对照优势。参数解释限定为数学作用及现有控制试验中的响应趋势，不虚构由比降和糙率推定k、n的通用关系。

正式回复每条采用原始意见、事实性答复、具体修改位置，未完成条目不得改成“已完成”。正文保留论文的连贯论证，不把任务清单、操作说明和内部核查问题写进正式正文。

新公式保持Word可编辑，另保存LaTeX源；保留原稿公式的编号与上下文，最后统一新增公式编号。保留清洁稿和按编辑要求标明修改的版本。完成标准为13条均有真实处理结果、正文与回复互相对应、无占位符。
'''
taskd='''# 任务D 图表排版与投稿前核验

对应R2-4、R2-6、R2-7、R2-9。本轮已更新图4、图6的120 km刻度，扩展图7，并补图5的箱线及均值说明。新图中文宋体，英文、数字和希腊符号使用Times New Roman，保留矢量源与600 dpi TIFF。

最终合稿核对正文图号和数据来源，检查原稿旧图中单位是否需重绘。全文核对数值与单位半角空格、范围号、上下标、斜体、表6非线性K量纲和表1实际率定边界。新增公式不可转图片。

锁定页码后将每条回复的章节锚点补为准确页码及行号；本轮工作版为避免页码变动造成假定位，仅使用章节和段落锚点。逐页检查所有页面、图片和公式，比较原始包中未修改部件，核对红字与清洁稿内容一致。删除工作说明、待核项和提交前清单后，才可生成正式提交附件。11月10日为编辑部通知期限，工具不自动上传或投稿。
'''
for fn,s in [('任务A_基准复现与时滞对照.md',taska),('任务B_洪峰与机理分析.md',taskb),('任务C_论文和回复合稿.md',taskc),('任务D_图表与投稿核验.md',taskd)]: (BASE/'01_任务安排'/fn).write_text(s,encoding='utf-8')

for fn in ['analyze_revision.py','draw_revision.py','build_revision.py']:
    shutil.copy2(Path('work')/fn,BASE/'06_复现与核验'/fn)
shutil.copy2(Path('work/artifact.md'),BASE/'06_复现与核验/文档保留与修改约定.md')
oldparts={n:ZipFile(SRC).read(n) for n in ZipFile(SRC).namelist()}
oldroot=etree.fromstring(oldparts['word/document.xml']);oldps=oldroot.findall('.//w:p',NS)
display=[]
for i,p in enumerate(oldps[:-1]):
    if p.find('.//m:oMath',NS) is not None and re.fullmatch(r'\(\d+\)',textof(oldps[i+1])):
        display.extend([etree.tostring(x) for x in p.findall('.//m:oMath',NS)])
newmath=[etree.tostring(x) for x in root.findall('.//m:oMath',NS)]
audit={'original_sha256':hashlib.sha256(SRC.read_bytes()).hexdigest(),'modified_package_parts':[n for n in parts if parts[n]!=oldparts.get(n)],'original_math_count':len(oldroot.findall('.//m:oMath',NS)),'revised_math_count':len(root.findall('.//m:oMath',NS)),'numbered_equations_found':len(display),'numbered_equations_byte_preserved':all(x in newmath for x in display),'new_formula_count':4,'working_comments':3,'reviewer_comments':13,'closed_or_text_revised':8,'partially_completed_or_pending':5,'title':'保留原题','source_files_unchanged':True,'rendering':'pending'}
(BASE/'06_复现与核验/文档结构核验.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding='utf-8')
print(MAN);print(RESP);print(json.dumps(audit,ensure_ascii=False))
