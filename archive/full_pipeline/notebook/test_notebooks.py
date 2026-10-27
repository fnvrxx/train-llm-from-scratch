"""Validate generated notebooks and execute them without access to repo imports."""
import ast
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from build_notebooks import OUT, build, extract

HARNESS = r'''
import builtins, io, json, math, os, sys
from pathlib import Path
import torch
from IPython.core.interactiveshell import InteractiveShell
from IPython.utils.capture import capture_output
payload = json.loads(sys.stdin.read())
# Initialize lazy optimizer dependencies before guarding notebook execution.
p = torch.nn.Parameter(torch.ones(1))
torch.optim.AdamW([p])
shell = InteractiveShell.instance()
original_import = builtins.__import__
def isolated_import(name, *args, **kwargs):
    if name.split('.')[0] in {'src', 'experiments', 'data_loader', 'config'}:
        raise AssertionError('Repository import: ' + name)
    return original_import(name, *args, **kwargs)
builtins.__import__ = isolated_import
# Deny writes by notebook code, while permitting read-only dependency imports.
def audit(event, args):
    if event == 'open':
        mode, flags = args[1:3]
        if (isinstance(mode, str) and any(c in mode for c in 'wax+')) or (isinstance(flags, int) and flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC)):
            raise AssertionError('Notebook attempted file write: ' + str(args[0]))
    if event in {'os.mkdir', 'os.rename', 'os.remove', 'os.rmdir', 'subprocess.Popen', 'os.system'}:
        raise AssertionError('Notebook attempted side effect: ' + event)
sys.addaudithook(audit)
for index, cell in enumerate(payload['cells']):
    if cell['cell_type'] != 'code':
        continue
    with capture_output() as output:
        result = shell.run_cell(cell['source'], store_history=False)
    if result.error_before_exec or result.error_in_exec:
        raise AssertionError(f'cell {index}: {output.stdout} {output.stderr}') from (result.error_before_exec or result.error_in_exec)
    if cell['metadata'].get('role') != 'definition':
        assert output.stdout or output.outputs, f'cell {index} has no visible output'
ns = shell.user_ns
assert len(ns['history']) == 10 and all(math.isfinite(x) for x in ns['history'])
assert ns['delta'] > 0
assert ns['generated'].shape == (1, 28)
assert ns['Transformer'].__module__ == '__main__'
assert torch.isfinite(ns['loss'])
assert all(torch.isfinite(p.grad).all() for p in ns['model'].parameters() if p.grad is not None)
kind = payload['metadata']['kind']
mlp, x = ns['mlp'], ns['mlp_input']
if kind == 'pruned':
    for layer, mask in [(mlp.hidden, mlp.hidden_mask), (mlp.proj, mlp.proj_mask)]:
        assert mask.sum() == mask.numel() // 2
        assert torch.count_nonzero(layer.weight.grad[mask == 0]) == 0
elif kind == 'topk':
    hidden = torch.relu(mlp.hidden(x))
    values, indices = hidden.topk(mlp.k, dim=-1)
    selected = torch.zeros_like(hidden).scatter(-1, indices, values)
    expected = torch.nn.functional.linear(selected, mlp.proj.weight, mlp.proj.bias)
    assert mlp.k == 16 and torch.allclose(mlp(x), expected, atol=1e-6)
elif kind == 'moe':
    assert torch.allclose(ns['gate_weights'].sum(-1), torch.ones(x.shape[0] * x.shape[1]))
    assert ns['chosen'].shape == (24, 2)
    dense = ns['MLP'](16)
    full = ns['MoEMLP'](dense, 4, 4)
    with torch.no_grad(): full.gate.weight.zero_()
    assert torch.allclose(full(x), dense(x), atol=1e-6)
    assert torch.isfinite(mlp.aux_loss)
elif kind == 'evolutionary':
    assert ns['active_before'] == ns['active_after'] and ns['grown'] > 0
    weight, mask = next(mlp.masked_layers())
    before = mask.clone()
    opt = ns['optimizer']
    old_step = opt.state[weight]['step'].clone()
    count = ns['rewire_weight'](weight, mask, opt, .3)
    changed = before != mask
    assert int(changed.sum()) == 2 * count and before.sum() == mask.sum()
    assert torch.count_nonzero(weight[changed]) == 0
    for value in opt.state[weight].values():
        if isinstance(value, torch.Tensor) and value.shape == weight.shape:
            assert torch.count_nonzero(value[changed]) == 0
    assert torch.equal(old_step, opt.state[weight]['step'])
    ns['enforce_model_masks'](ns['model'])
    assert torch.count_nonzero(weight[~mask]) == 0
else:
    assert ns['preference_accuracy'](ns['chosen_rewards'], ns['rejected_rewards']).item() == .5
    assert abs(ns['advantages'].mean().item()) < 1e-6
    assert all(torch.isfinite(ns[key]) for key in ('supervised_loss', 'preference_loss', 'preference_dpo', 'ppo_demo', 'grpo_demo'))
    equal = torch.zeros(2, 3)
    assert torch.equal(ns['k3_kl'](equal, equal), equal)
    logits = torch.randn(1, 4, 5, requires_grad=True)
    loss = ns['sft_loss'](logits, torch.tensor([[0,1,2,3]]), torch.tensor([[0,0,0,1]]))
    loss.backward()
    assert torch.count_nonzero(logits.grad[:, :2]) == 0
    assert torch.count_nonzero(logits.grad[:, 2]) > 0
print(kind, 'OK: 10 langkah, gradien finite, generasi, invariant sparse/objective, tanpa artifact')
'''


class NotebookTests(unittest.TestCase):
    def test_structure_and_source(self):
        import nbformat
        for path in sorted(OUT.glob('*.ipynb')):
            with self.subTest(notebook=path.name):
                payload = json.loads(path.read_text())
                nbformat.validate(nbformat.from_dict(payload))
                self.assertEqual(payload, build(payload['metadata']['kind']))
                for index, cell in enumerate(payload['cells']):
                    if cell['cell_type'] != 'code':
                        continue
                    self.assertEqual(cell['outputs'], [])
                    self.assertIsNone(cell['execution_count'])
                    self.assertIn('AntiVibe', payload['cells'][index - 1]['source'])
                    tree = ast.parse(cell['source'])
                    for node in ast.walk(tree):
                        if isinstance(node, (ast.Import, ast.ImportFrom)):
                            names = [node.module] if isinstance(node, ast.ImportFrom) else [a.name for a in node.names]
                            self.assertTrue(all(n.split('.')[0] in {'torch', 'math', 'copy'} for n in names))
                        if isinstance(node, ast.Call):
                            name = node.func.id if isinstance(node.func, ast.Name) else getattr(node.func, 'attr', '')
                            self.assertNotIn(name, {'exec', 'open', 'save', 'load', 'mkdir', 'makedirs', 'write_text', 'write_bytes', 'system', 'run', 'Popen'})
                            if isinstance(node.func, ast.Name):
                                self.assertNotEqual(name, 'eval')
                    metadata = cell['metadata']
                    if 'source_path' in metadata:
                        expected = next(extract(metadata['source_path'], [metadata['source_symbol']]))
                        self.assertEqual(cell['source'], expected['source'])

    def test_run_in_isolation(self):
        for path in sorted(OUT.glob('*.ipynb')):
            with self.subTest(notebook=path.name), tempfile.TemporaryDirectory() as directory:
                env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', IPYTHONDIR=directory, MPLCONFIGDIR=directory)
                env.pop('PYTHONPATH', None)
                result = subprocess.run([sys.executable, '-I', '-B', '-c', HARNESS],
                                        input=path.read_text(), text=True, capture_output=True,
                                        cwd=directory, env=env, timeout=120)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                print(result.stdout.strip())


if __name__ == '__main__':
    unittest.main()
