"""Retreina a implementação preservada em splits fixos, com manifesto explícito."""
import argparse
import contextlib
import csv
import hashlib
import importlib.metadata
import json
import os
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pipeline', type=Path, required=True)
    parser.add_argument('--site-packages', type=Path, required=True)
    parser.add_argument('--source-run', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--seeds', type=int, nargs='+', default=[42,43,44])
    args = parser.parse_args()
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    sys.path.insert(0, str(args.site_packages))
    sys.path.insert(0, str(args.pipeline))
    import torch
    from src.bert_attention.train import train
    from src.evaluate.select_final_models import select, evaluate_test
    assert torch.backends.mps.is_available(), 'A rodada foi planejada para MPS; nenhuma troca silenciosa para CPU.'
    versions = {name: importlib.metadata.version(name) for name in ['torch','transformers','numpy','pandas','scikit-learn','safetensors']}
    source = args.source_run / 'prepared'
    for seed in args.seeds:
        run = args.output_root / ('seed_' + str(seed))
        if run.exists():
            raise FileExistsError(f'Não sobrescrever rodada existente: {run}')
        (run / 'prepared').mkdir(parents=True)
        (run / 'logs').mkdir()
        for split in ('train','val','test'):
            shutil.copyfile(source / (split+'.csv'), run/'prepared'/(split+'.csv'))
        split_rows = {}
        for split in ('train','val','test'):
            with (run/'prepared'/(split+'.csv')).open(newline='') as handle:
                split_rows[split] = list(csv.DictReader(handle))
        for first, second in [('train','val'),('train','test'),('val','test')]:
            for field in ('edition_id','work_id'):
                left = {r.get(field,'') for r in split_rows[first]} - {''}
                right = {r.get(field,'') for r in split_rows[second]} - {''}
                assert not left & right
        values = dict(seed=seed, epochs=60, patience=8, lr=0.0005,
                      weight_decay=0.01, batch_size=16, use_weights=True,
                      unfreeze_last=0, evaluate_test=False, run_dir=str(run))
        manifest = dict(audit_date='2026-11-22', started_at_utc=datetime.now(timezone.utc).isoformat(),
                        source_code_commit='564ef1aaaf9ec72f15797459a29f4d3b8d6c160b',
                        purpose='Retreinamento de auditoria em partições atuais; não é novo teste independente.',
                        arguments={k:v for k,v in values.items() if k!='run_dir'},
                        versions=versions, device='mps', max_tokens=128,
                        split_counts={s:len(r) for s,r in split_rows.items()},
                        split_sha256={s:digest(run/'prepared'/(s+'.csv')) for s in split_rows},
                        source_code_sha256={name:digest(args.pipeline/name) for name in
                                            ['src/bert_attention/model.py','src/bert_attention/train.py','src/evaluate/select_final_models.py']},
                        checkpoint_selection='Maior acurácia de validação; primeiro checkpoint nos empates; nenhuma leitura do teste durante treino/seleção.',
                        status='training')
        def save():
            (run/'training_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
        save()
        started=time.monotonic()
        print(f'INÍCIO seed={seed} | MPS | máximo60épocas/paciência8',flush=True)
        try:
            with (run/'logs/training.log').open('w',buffering=1) as log:
                with contextlib.redirect_stdout(log),contextlib.redirect_stderr(log):
                    train(argparse.Namespace(**values))
            manifest['status']='validation_selection'
            manifest['training_seconds']=round(time.monotonic()-started,3)
            save()
            print(f'TREINO CONCLUÍDO seed={seed}; seleção na validação',flush=True)
            checkpoint=run/'models/bert_attention'
            with (run/'logs/selection.log').open('w',buffering=1) as log:
                with contextlib.redirect_stdout(log),contextlib.redirect_stderr(log):
                    validation=select(run/'prepared',run/'results',checkpoint,16)
            manifest['status']='test_evaluation'
            manifest['selection'] = json.loads((run/'results/selecao_final.json').read_text())
            save()
            with (run/'logs/test.log').open('w',buffering=1) as log:
                with contextlib.redirect_stdout(log),contextlib.redirect_stderr(log):
                    test=evaluate_test(run/'prepared',run/'results',16)
            manifest.update(status='complete',finished_at_utc=datetime.now(timezone.utc).isoformat(),
                            total_seconds=round(time.monotonic()-started,3),
                            checkpoint_sha256=digest(checkpoint/'model.pt'),
                            test_summary={m:dict(accuracy=r['accuracy_por_nivel']['full']['accuracy'],macro_f1=r['macro_f1']) for m,r in test.items()})
            save()
            print(json.dumps(dict(seed=seed,status='complete',seconds=manifest['total_seconds'],test=manifest['test_summary']),ensure_ascii=False),flush=True)
        except BaseException as exc:
            manifest.update(status='failed',error=repr(exc),seconds=round(time.monotonic()-started,3))
            save()
            raise
        torch.mps.empty_cache()


if __name__=='__main__':
    main()
