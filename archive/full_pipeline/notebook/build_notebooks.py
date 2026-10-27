"""Build self-contained teaching notebooks; no generated cell writes artifacts."""
import ast
import json
from pathlib import Path
from textwrap import dedent

from learning_cells import learning_cells, explain_code, concept, PRACTICE

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent
BASE = [('src/models/mlp.py', ['MLP']),
        ('src/models/attention.py', ['Head', 'MultiHeadAttention']),
        ('src/models/transformer_block.py', ['Block']),
        ('src/models/transformer.py', ['Transformer'])]
POST = [('utils', ['masked_mean']), ('sft', ['sft_loss']),
        ('reward_train', ['bradley_terry_loss', 'preference_accuracy']),
        ('dpo', ['dpo_loss']), ('ppo', ['ppo_policy_loss']),
        ('grpo', ['group_advantages', 'k3_kl', 'grpo_loss'])]
VARIANTS = {'baseline': [], 'pruned': ['PrunedMLP', 'convert_model'],
            'topk': ['TopKMLP', 'convert_model'],
            'moe': ['Expert', 'MoEMLP', 'convert_model'],
            'evolutionary': ['EvolutionaryMLP', 'convert_model', 'evolutionary_mlps']}


def markdown(source):
    return dict(cell_type='markdown', metadata={}, source=dedent(source).strip() + '\n')


def code(source, **metadata):
    source = dedent(source).strip() + '\n'
    ast.parse(source)
    return dict(cell_type='code', metadata=metadata, source=source,
                outputs=[], execution_count=None)


def extract(path, names):
    source = (ROOT / path).read_text()
    nodes = {node.name: node for node in ast.parse(source).body
             if isinstance(node, (ast.ClassDef, ast.FunctionDef))}
    for name in names:
        node = nodes[name]
        start = min([node.lineno] + [d.lineno for d in node.decorator_list])
        yield code('\n'.join(source.splitlines()[start - 1:node.end_lineno]),
                   source_path=path, source_symbol=name, role='definition')


def build(kind):
    cells = [markdown(f'''# Memahami LLM melalui kode: {kind}

Notebook mandiri untuk pembaca dengan dasar matematika. Jalankan dari atas ke bawah
pada kernel Python dengan PyTorch terpasang. CPU cukup. Semua data, bobot, dan hasil
berada di memori; hasil muncul di bawah sel ketika dijalankan. Sel definisi memang
belum menghasilkan output. Notebook yang dibagikan menyimpan output kosong.

Baca urutannya: definisi model, data karakter, attention, loss, gradien, sepuluh
langkah latihan, lalu generasi. Buka tabel AntiVibe di dekat kode untuk melacak nama
variabel. Jika ada perubahan definisi, restart kernel lalu Run all.

$B$ = jumlah contoh, $T$ = panjang urutan, $C$ = dimensi embedding,
$V$ = banyak token, $H=4C$ = lebar MLP. Tensor adalah array multidimensi;
`[B,T,C]` dapat dipandang sebagai $BT$ vektor di $\\mathbb{{R}}^C$.

Referensi konteks: [tutorial penulis](https://fareedkhan-dev.github.io/train-llm-from-scratch/).
Kode berikut diambil dari repositori lokal; varian sparse merupakan eksperimen repo ini.
'''), code('''
import copy
import math
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint

torch.set_num_threads(1)
print('PyTorch:', torch.__version__, '| laboratorium CPU')
''', role='setup')]
    for path, names in BASE:
        cells.extend(extract(path, names))
    if kind != 'baseline':
        cells.extend(extract(f'experiments/{kind}/model.py', VARIANTS[kind]))
    if kind == 'evolutionary':
        cells.extend(extract('experiments/evolutionary/rewire.py',
                            ['_clear_optimizer_state', 'rewire_weight', 'rewire_model',
                             'enforce_model_masks', 'active_connections']))
    lessons = learning_cells(kind, markdown, code)
    for cell in lessons:
        if cell['cell_type'] == 'markdown' and '### Dari pretraining' in cell['source']:
            for module, names in POST:
                cells.extend(extract(f'src/post_training/{module}.py', names))
        cells.append(cell)
        if cell['cell_type'] == 'markdown':
            for prefix, lesson in PRACTICE.items():
                if prefix in cell['source']:
                    cells.append(markdown(lesson))
    enriched = []
    for cell in cells:
        if cell['cell_type'] == 'code':
            symbol = cell['metadata'].get('source_symbol')
            if symbol:
                enriched.append(markdown(concept(symbol, kind)))
            enriched.append(markdown(explain_code(cell['source'])))
        enriched.append(cell)
    for index, cell in enumerate(enriched):
        cell['id'] = f'{kind}-{index:03d}'
    return dict(cells=enriched, nbformat=4, nbformat_minor=5,
                metadata=dict(kernelspec=dict(display_name='Python 3', language='python', name='python3'),
                              language_info=dict(name='python', version='3.10'), kind=kind))


def main():
    for i, kind in enumerate(VARIANTS, 1):
        name = 'evolutionary_set' if kind == 'evolutionary' else kind
        payload = build(kind)
        (OUT / f'{i:02d}_{name}.ipynb').write_text(json.dumps(payload, ensure_ascii=False, indent=1) + '\n')
        print(f'{name}: {len(payload["cells"])} sel')


if __name__ == '__main__':
    main()
