#!/usr/bin/env python3
"""Sample one published clean-condition error per representative model/task family.

Uses only the checksum-pinned release, not model inference. For each family,
maximize mean clean accuracy over the published shot sweep (ties: fewer shots),
then sample uniformly from its incorrect trial records with an independent,
fixed seed per cell. No errors are filtered for visual appeal or error type.

Example:
  python scripts/analysis/sample_mistakes_t128.py --seed 20260909 \
    --publication-revision 17640b7b588090dc62b944e47c79c1b885d989c2
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import random
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MODELS = (
    ('qwen3', 'Language', 'Qwen3-14B'),
    ('evo2', 'Genome', 'Evo2-40B'),
    ('nextterm', 'Integer seq.', 'NextTerm-440M'),
    ('imagegpt', 'Image', 'ImageGPT-large'),
    ('timesfm', 'Time series', 'TimesFM 2.5'),
    ('progen2', 'Protein', 'ProGen2-base'),
)
FAMILIES = ('Global Broadcast', 'Half Segments', 'Parity Masking',
            'Half Swaps', 'Bit Inversion', 'Bit Reversal', 'Local Shifts')
SOURCE_FILES = (
    'configs/paper_runs_t128.json', 'data/t128/artifact_manifest.json',
    'data/t128/raw_results.sha256', 'data/t128/canonical_semantic_clusters_k7.json',
    'data/t128/model_revisions.json', 'src/dsl.py',
    'src/program_synth.py', 'src/program_synth_evo_smartbatch.py',
    'src/curated_transformations.jsonl',
)
spec = importlib.util.spec_from_file_location('published_dsl', ROOT / 'src/dsl.py')
dsl = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dsl)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def published_sources(revision: str) -> dict:
    """Fail if the files used locally differ from the named published commit."""
    hashes = {}
    for name in SOURCE_FILES:
        published = subprocess.check_output(['git', 'show', f'{revision}:{name}'], cwd=ROOT)
        current = (ROOT / name).read_bytes()
        if current != published:
            raise ValueError(f'{name} differs from published revision {revision}')
        hashes[name] = hashlib.sha256(current).hexdigest()
    return hashes


def load_clusters() -> dict[int, set[tuple[str, ...]]]:
    data = json.loads((ROOT / 'data/t128/canonical_semantic_clusters_k7.json').read_text())
    clusters = {c['cluster_id']: {tuple(f['program']) for f in c['functions']}
                for c in data['clusters']}
    assert sorted(clusters) == list(range(1, 8))
    assert sum(map(len, clusters.values())) == len(set.union(*clusters.values())) == 100
    return clusters


def verify_cell(data: dict, shot: int, expected_programs: set) -> dict:
    config = data['config']
    assert config['in_context_examples'] == shot
    assert config['trials_per_program'] == 128
    assert config['bit_length'] == 8
    assert config['ablate_labels'] is False
    tasks = {tuple(t['program']): t for t in data['tasks']}
    assert len(data['tasks']) == len(tasks) == 100
    assert set(tasks) == expected_programs
    for task in tasks.values():
        assert len(task['trials']) == task['total_trials'] == 128
        for trial in task['trials']:
            assert isinstance(trial['correct'], bool)
            assert isinstance(trial['prediction'], str)
            assert re.fullmatch('[01]{8}', trial['expected'])
            assert trial['correct'] == (trial['expected'] == trial['prediction'])
        assert sum(t['correct'] for t in task['trials']) == task['correct']
        assert abs(task['accuracy'] - task['correct'] / 128) < 1e-12
    return tasks


def recover_input(run: str, program: tuple[str, ...], trial: dict, shot: int) -> tuple[str, dict]:
    """Uniquely recover bits from the gold query and every clean demonstration.

    Testing only f(query) against expected is ambiguous for complement-invariant
    functions. Testing every encoded demonstration under both symbol assignments
    resolves that ambiguity without consulting the model's prediction.
    """
    expected = trial['expected']
    context_value = expected if program == ('meta_constant',) else None
    f = lambda bits: dsl.run_program(program, bits, s_m1=context_value)
    prompt, query = trial['prompt'], trial['query']
    if run in ('imagegpt', 'timesfm'):
        assert re.fullmatch('[01]{8}', query)
        if run == 'imagegpt':
            pairs = re.findall(r'^([01]{8})->([01]{8})$', prompt, re.M)
        else:
            pairs = re.findall(r'^([01]{8}) SEP ([01]{8}) SEP$', prompt, re.M)
        assert len(pairs) == shot
        assert all(f(i) == o for i, o in pairs)
        assert f(query) == expected
        return query, {'method': 'stored_binary_query', 'demonstrations_checked': shot}

    if run == 'nextterm':
        chunks = prompt.split(',')
        assert chunks[-1] == ''
        chunks.pop()
        assert len(chunks) == 2 * shot + 1 and chunks[-1] == query
        pairs = list(zip(chunks[:-1:2], chunks[1:-1:2]))
    else:
        if run == 'progen2':
            assert prompt[0] == '1'  # published ProGen2 sequence-start token
            prompt = prompt[1:]
        assert len(prompt) == 17 * shot + 8 and prompt[-8:] == query
        pairs = [(prompt[17*i:17*i+8], prompt[17*i+8:17*i+16]) for i in range(shot)]
        separators = {prompt[17*i+16] for i in range(shot)}
        assert len(separators) == 1
    symbols = sorted(set(query + ''.join(a+b for a,b in pairs)))
    assert len(symbols) == 2, symbols
    if run != 'nextterm':
        assert not separators.intersection(symbols)
    solutions = []
    for bits in ('01', '10'):
        mapping = dict(zip(symbols, bits))
        decode = lambda s: ''.join(mapping[c] for c in s)
        q = decode(query)
        if f(q) == expected and all(f(decode(i)) == decode(o) for i, o in pairs):
            solutions.append((q, mapping))
    assert len(solutions) == 1, (run, program, query, expected, solutions)
    q, mapping = solutions[0]
    return q, {'method': 'unique_symbol_map_validated_on_all_clean_demonstrations_and_gold_query',
               'symbol_to_bit': mapping, 'demonstrations_checked': shot}


def tex_escape(s: str) -> str:
    mapping = {'\\': r'\textbackslash{}', '_': r'\_', '%': r'\%', '&': r'\&',
               '#': r'\#', '{': r'\{', '}': r'\}', '$': r'\$', '^': r'\textasciicircum{}', '~': r'\textasciitilde{}'}
    return ''.join(mapping.get(c, c) for c in s)


def latex_table(rows: list[dict]) -> str:
    out = [r'\resizebox{\textwidth}{!}{%', r'\begin{tabular}{lccccccc}', r'\toprule',
           ' & ' + ' & '.join(r'\shortstack{' + n.replace(' ', r'\\ ') + '}' for n in FAMILIES) + r' \\', r'\midrule']
    for run, modality, model in MODELS:
        cells = []
        for family in FAMILIES:
            row = next(r for r in rows if r['run_key'] == run and r['family'] == family)
            parts = row['program'].split('+')
            program_lines = [r'{\tiny '+tex_escape(parts[0])+r'}']
            if len(parts) > 1:
                program_lines.append(r'{\tiny \ \ +'+tex_escape('+'.join(parts[1:]))+f", $n{{=}}{row['best_clean_shot']}$"+'}')
            else:
                program_lines[0] = program_lines[0][:-1]+f", $n{{=}}{row['best_clean_shot']}$"+'}'
            for label,key in [('in','input'),('exp','expected'),('out','prediction')]:
                value = r'\texttt{'+tex_escape(row[key])+'}' if row[key] else r'$\varnothing$'
                program_lines.append(r'{\tiny '+label+'}~'+value)
            cells.append(r'\begin{tabular}[c]{@{}l@{}}'+r'\\'.join(program_lines)+r'\end{tabular}')
        out.append(r'\shortstack[l]{'+tex_escape(modality)+r'\\ {\tiny ('+tex_escape(model)+')}} & '+' & '.join(cells)+r' \\[14pt]')
    out.extend([r'\bottomrule',r'\end{tabular}}'])
    return '\n'.join(out)+'\n'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, default=20260909)
    parser.add_argument('--publication-revision', default='17640b7b588090dc62b944e47c79c1b885d989c2')
    parser.add_argument('--results-root', type=Path, default=ROOT/'data/t128/results_128')
    parser.add_argument('--output-dir', type=Path, default=ROOT/'results/t128/analysis/paper')
    parser.add_argument('--previous-csv', type=Path, help='Optional audit only: fail if any displayed example is repeated; does not alter sampling.')
    args = parser.parse_args()
    source_hashes = published_sources(args.publication_revision)
    clusters = load_clusters()
    programs = set.union(*clusters.values())
    cfg = json.loads((ROOT/'configs/paper_runs_t128.json').read_text())
    artifact = json.loads((ROOT/'data/t128/artifact_manifest.json').read_text())
    hashes = {}
    for line in (ROOT/'data/t128/raw_results.sha256').read_text().splitlines():
        digest,name = line.split(maxsplit=1)
        hashes[name.split('/results_128/',1)[1]] = digest
    rows,profiles,used_hashes = [],[],{}
    predictions = 0
    for run,modality,model in MODELS:
        condition = cfg['runs'][run]['conditions']['clean']
        best = {}
        for shot in sorted(condition['shots']):
            relative = condition['output_template'].format(shot=shot)
            path = args.results_root/relative
            digest = sha256(path)
            assert digest == hashes[relative], f'Hash mismatch: {path}'
            used_hashes[relative] = digest
            data = json.loads(path.read_text())
            tasks = verify_cell(data,shot,programs)
            predictions += 12800
            for cid,family_programs in clusters.items():
                correct = sum(tasks[p]['correct'] for p in family_programs)
                total = len(family_programs)*128
                profiles.append(dict(run_key=run,model=model,family=FAMILIES[cid-1],shot=shot,
                                     correct=correct,total=total,accuracy=correct/total))
                if cid not in best or correct > best[cid]['correct']:
                    best[cid] = dict(shot=shot,correct=correct,total=total,path=relative)
            print(f'verified {model}: n={shot}, 100 tasks × 128 trials',flush=True)
            del tasks,data
        # Reread only winning files; never retain all raw sweeps in memory.
        cache = {}
        for cid in sorted(clusters):
            winner = best[cid]
            relative = winner['path']
            if relative not in cache:
                cache[relative] = json.loads((args.results_root/relative).read_text())
            tasks = cache[relative]['tasks']
            pool = [(task_pos,trial_pos) for task_pos,t in enumerate(tasks)
                    if tuple(t['program']) in clusters[cid]
                    for trial_pos,tr in enumerate(t['trials']) if not tr['correct']]
            assert len(pool) == winner['total']-winner['correct'] > 0
            cell_seed = f'{args.seed}:{run}:{cid}'
            draw = random.Random(cell_seed).randrange(len(pool))
            task_pos,trial_pos = pool[draw]
            task = tasks[task_pos];trial = task['trials'][trial_pos]
            query,decoding = recover_input(run,tuple(task['program']),trial,winner['shot'])
            row = dict(modality=modality,family=FAMILIES[cid-1],best_clean_shot=winner['shot'],
                       family_accuracy=winner['correct']/winner['total'],program='+'.join(task['program']),
                       input=query,expected=trial['expected'],prediction=trial['prediction'],
                       model=model,run_key=run,source_file=relative,source_sha256=hashes[relative],
                       task_index=task['index'],task_array_index=task_pos,trial_index=trial_pos,
                       eligible_incorrect_trials=len(pool),sampled_pool_index=draw,cell_seed=cell_seed,
                       query_encoded=trial['query'],decoding=decoding,
                       trial_sha256=hashlib.sha256(json.dumps(trial,sort_keys=True,separators=(',',':')).encode()).hexdigest())
            rows.append(row)
        del cache
        print(f'sampled all seven families for {model}',flush=True)
    assert len(rows) == 42 and predictions == 537600
    prior_count = None
    if args.previous_csv:
        old = list(csv.DictReader(args.previous_csv.open()))
        keys = ('modality','family','program','input','expected','prediction')
        previous = {tuple(r[k] for k in keys) for r in old}
        repeated = [r for r in rows if tuple(str(r[k]) for k in keys) in previous]
        assert not repeated, f'{len(repeated)} samples repeat the previous display; no alternate sample was chosen.'
        prior_count = len(old)
    args.output_dir.mkdir(parents=True,exist_ok=True)
    csv_fields = [k for k in rows[0] if k!='decoding']
    with (args.output_dir/'mistake_matrix.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=csv_fields,extrasaction='ignore',lineterminator='\n');writer.writeheader();writer.writerows(rows)
    (args.output_dir/'mistake_matrix.tex').write_text(latex_table(rows))
    with (args.output_dir/'mistake_family_shot_profiles.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(profiles[0]),lineterminator='\n');writer.writeheader();writer.writerows(profiles)
    audit = dict(schema_version=1,publication_repository='https://github.com/N8python/shared-emergence-icl-modalities',
                 publication_revision=args.publication_revision,artifact=artifact,source_hashes=source_hashes,
                 sampler_sha256=sha256(Path(__file__)),seed=args.seed,selection='Maximum family mean clean accuracy, ties to smaller shot count; uniform draw from all incorrect trial records with per-cell seed.',
                 condition='clean',trials_per_task=128,validated_clean_cells=len(used_hashes),
                 validated_predictions=predictions,used_result_hashes=used_hashes,
                 previous_examples_checked=prior_count,previous_display_repeats=0 if prior_count is not None else None,
                 index_convention='task_array_index and trial_index are zero-based',rows=rows)
    (args.output_dir/'mistake_matrix_provenance.json').write_text(json.dumps(audit,indent=2)+'\n')
    print(f'Wrote 42 sampled mistakes from {len(used_hashes)} published clean cells ({predictions:,} predictions).',flush=True)
    if prior_count is not None:print(f'All 42 displays differ from the {prior_count} previous examples.',flush=True)


if __name__ == '__main__':
    main()
