"""Direct-authored bitmap alphabet expansion; no font or dataset weight fitting.

The inherited digit head is retained. New 5x7 stroke patterns are literal
author choices expanded to15x12. This is a stylized hypothesis, not a claim
of scanned-document recognition or independent accuracy.
"""
import hashlib
import copy
import json
from pathlib import Path
import numpy as np
from cellforge.ir.graph import Graph
from cellforge.construction.transactions import apply_ops
from cellforge.governance.ledger import Ledger
from cellforge.runtime.interp import Interpreter
from cellforge.runtime.vision import OCRPredictor
from cellforge.data.render import render_one
from PIL import ImageFont

# Each slash-separated row is a literal five-bit stroke design.
GLYPHS = {
 'A':'01110/10001/10001/11111/10001/10001/10001',
 'B':'11110/10001/10001/11110/10001/10001/11110',
 'C':'01111/10000/10000/10000/10000/10000/01111',
 'D':'11110/10001/10001/10001/10001/10001/11110',
 'E':'11111/10000/10000/11110/10000/10000/11111',
 'F':'11111/10000/10000/11110/10000/10000/10000',
 'G':'01111/10000/10000/10111/10001/10001/01111',
 'H':'10001/10001/10001/11111/10001/10001/10001',
 'I':'11111/00100/00100/00100/00100/00100/11111',
 'J':'00111/00010/00010/00010/00010/10010/01100',
 'K':'10001/10010/10100/11000/10100/10010/10001',
 'L':'10000/10000/10000/10000/10000/10000/11111',
 'M':'10001/11011/10101/10101/10001/10001/10001',
 'N':'10001/11001/10101/10011/10001/10001/10001',
 'O':'01110/10001/10001/10001/10001/10001/01110',
 'P':'11110/10001/10001/11110/10000/10000/10000',
 'Q':'01110/10001/10001/10001/10101/10010/01101',
 'R':'11110/10001/10001/11110/10100/10010/10001',
 'S':'01111/10000/10000/01110/00001/00001/11110',
 'T':'11111/00100/00100/00100/00100/00100/00100',
 'U':'10001/10001/10001/10001/10001/10001/01110',
 'V':'10001/10001/10001/10001/10001/01010/00100',
 'W':'10001/10001/10001/10101/10101/10101/01010',
 'X':'10001/10001/01010/00100/01010/10001/10001',
 'Y':'10001/10001/01010/00100/00100/00100/00100',
 'Z':'11111/00001/00010/00100/01000/10000/11111',
 'a':'00000/00000/01110/00001/01111/10001/01111',
 'b':'10000/10000/10110/11001/10001/10001/11110',
 'c':'00000/00000/01111/10000/10000/10000/01111',
 'd':'00001/00001/01101/10011/10001/10001/01111',
 'e':'00000/00000/01110/10001/11111/10000/01111',
 'f':'00110/01001/01000/11100/01000/01000/01000',
 'g':'00000/01111/10001/10001/01111/00001/01110',
 'h':'10000/10000/10110/11001/10001/10001/10001',
 'i':'00100/00000/01100/00100/00100/00100/01110',
 'j':'00010/00000/00110/00010/00010/10010/01100',
 'k':'10000/10000/10010/10100/11000/10100/10010',
 'l':'01100/00100/00100/00100/00100/00100/01110',
 'm':'00000/00000/11010/10101/10101/10101/10101',
 'n':'00000/00000/10110/11001/10001/10001/10001',
 'o':'00000/00000/01110/10001/10001/10001/01110',
 'p':'00000/11110/10001/10001/11110/10000/10000',
 'q':'00000/01111/10001/10001/01111/00001/00001',
 'r':'00000/00000/10110/11001/10000/10000/10000',
 's':'00000/00000/01111/10000/01110/00001/11110',
 't':'01000/01000/11100/01000/01000/01001/00110',
 'u':'00000/00000/10001/10001/10001/10011/01101',
 'v':'00000/00000/10001/10001/10001/01010/00100',
 'w':'00000/00000/10001/10001/10101/10101/01010',
 'x':'00000/00000/10001/01010/00100/01010/10001',
 'y':'00000/10001/10001/10001/01111/00001/01110',
 'z':'00000/00000/11111/00010/00100/01000/11111',
 '.':'00000/00000/00000/00000/00000/00110/00110',
 ',':'00000/00000/00000/00000/00110/00110/00100',
 ':':'00000/00110/00110/00000/00110/00110/00000',
 ';':'00000/00110/00110/00000/00110/00110/00100',
 '-':'00000/00000/00000/11111/00000/00000/00000',
 '!':'00100/00100/00100/00100/00100/00000/00100',
 '?':'01110/10001/00001/00010/00100/00000/00100',
 '/':'00001/00001/00010/00100/01000/10000/10000',
 ' ':'00000/00000/00000/00000/00000/00000/00000',
}


def patch(character):
    pattern = np.array([[int(bit) for bit in row] for row in GLYPHS[character].split('/')], dtype=float)
    if pattern.shape != (7,5):
        raise ValueError('glyph literal must be7x5')
    pixels = np.zeros((15,12))
    pixels[:14,1:11] = np.repeat(np.repeat(pattern,2,0),2,1)
    return pixels


def build(parent):
    chars = list(GLYPHS)
    weights, biases = [], []
    for char in chars:
        p = patch(char).reshape(-1)
        weights.append(2*(2*p-1)/180)
        biases.append((180-2*p.sum())/180)
    ops = [{'op':'set_param','name':'glyph.W1','value':np.concatenate([parent.params['glyph.W1'],weights]).tolist()},
           {'op':'set_param','name':'glyph.b1','value':np.concatenate([parent.params['glyph.b1'],biases]).tolist()}]
    # EOS checks all remaining ink, so an interior blank can decode as space.
    eos = copy.deepcopy(parent.defs['eos_head'])
    eos['nodes'] = [node for node in eos['nodes'] if node['id'] not in ('nd','b','sb')]
    for node in eos['nodes']:
        if node['id'] == 'm':
            node['in'] = {'a':'sa','b':'sa'}
    ops.append({'op':'put_def','name':'eos_head','def':eos})
    child,_ = apply_ops(parent,ops)
    child.meta['vocab'] = parent.meta['vocab'] + chars
    child.meta['alphabet_hypothesis'] = 'direct-authored5x7-strokes/1'
    child.meta['max_len'] = 16
    child.validate()
    return child, ops


def main():
    output = Path('runs/parallel-vision/alphabet')
    parent = Graph.load('runs/p1/exports/r1-iter-a')
    child, ops = build(parent)
    child.save(output/'candidate')
    proposal = {'protocol':'authored-alphabet-expansion/1','parent_hash':parent.hash(),'child_hash':child.hash(),
                'ops':ops,'metadata_changes':{'vocab':child.meta['vocab'],'max_len':16,'alphabet_hypothesis':child.meta['alphabet_hypothesis']},
                'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                'lineage':'digit head/alignment inherited iter-a; literal alphabet weights and suffix-ink EOS authored here',
                'limits':'5x7 stylized bitmap hypothesis; no font fitting; no independent letter accuracy established; line width128 and page tiling limitations remain'}
    (output/'proposal.json').write_text(json.dumps(proposal,indent=2)+'\n')
    Ledger(output/'ledger.jsonl').append('authored_alphabet_proposal',proposal,'ocr-author')
    # Fixed development probes are evidence only, never inputs to weight creation.
    chars = list(GLYPHS)
    logits = Interpreter(child).run_cell('glyph_cell',np.stack([patch(c).ravel() for c in chars]))
    predictions = [child.meta['vocab'][3+int(i)] for i in logits.argmax(-1)]
    font_path = Path('fonts/DejaVuSansMono.ttf')
    font = ImageFont.truetype(str(font_path),20)
    ocr = OCRPredictor(child)
    probes = [{'source':text,'observation':ocr(render_one(text,font,6,0))}
              for text in ['A','abc','Hi','42','A B','hello']]
    report = {'graph_hash':child.hash(),
              'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'glyph_literal_self_consistency':[{'source':c,'prediction':p} for c,p in zip(chars,predictions)],
              'self_consistency_not_independent_benchmark':True,
              'independently_rendered_dev_font_probes':probes,
              'font_sha256':hashlib.sha256(font_path.read_bytes()).hexdigest(),
              'no_final_access':True,'weight_generation_reads_font_or_probe_data':False}
    (output/'development-probe.json').write_text(json.dumps(report,indent=2)+'\n')
    Ledger(output/'ledger.jsonl').append('alphabet_development_probe',report,'ocr-author')
    print(child.hash())


if __name__ == '__main__':
    main()
