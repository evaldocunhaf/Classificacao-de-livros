"""Compara blocos adicionais de checkpoints PyTorch ZIP sem importar torch.

O unpickler aceita apenas reconstruções de tensores/metadados de OrderedDict;
nenhuma classe arbitrária do pickle é importada ou executada.
"""
import argparse
import csv
import hashlib
import io
import json
import math
import pickle
import zipfile
from collections import OrderedDict
from pathlib import Path


def tensor(storage, offset, shape, stride, *rest):
    return dict(storage=storage, offset=offset, shape=shape, stride=stride)


class MetadataUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if (module, name) == ('collections', 'OrderedDict'):
            return OrderedDict
        if module == 'torch._utils' and name in ('_rebuild_tensor_v2', '_rebuild_tensor'):
            return tensor
        if module == 'torch' and name.endswith('Storage'):
            return name
        if module == 'torch._utils' and name == '_rebuild_parameter':
            return lambda data, *args: data
        raise ValueError(f'Unallowed pickle class: {module}.{name}')

    def persistent_load(self, value):
        assert value[0] == 'storage'
        return dict(dtype=value[1], key=str(value[2]), location=value[3], count=value[4])


def signature(path):
    with zipfile.ZipFile(path) as archive:
        pickled = next(n for n in archive.namelist() if n.endswith('/data.pkl'))
        prefix = pickled[:-len('data.pkl')]
        state = MetadataUnpickler(io.BytesIO(archive.read(pickled))).load()
        params = {}
        for key, value in state.items():
            if key.startswith('backbone.'):
                continue
            storage = value['storage']
            assert value['offset'] == 0
            assert math.prod(value['shape']) == storage['count']
            raw = archive.read(prefix + 'data/' + storage['key'])
            params[key] = dict(shape=value['shape'], dtype=storage['dtype'], sha256=hashlib.sha256(raw).hexdigest())
        return params


def read_csv(path):
    with path.open(encoding='utf-8', newline='') as handle:
        return list(csv.DictReader(handle))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pipeline', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    pipeline = args.pipeline.resolve()
    root = pipeline / 'output'
    final = root / 'runs/2026-10-16-final-comparison-subjects'
    actual = {s: read_csv(final / 'prepared' / (s+'.csv')) for s in ('train','val','test')}
    final_checkpoint = final / 'models/bert_attention/model.pt'
    final_signature = signature(final_checkpoint)
    reports = []
    for checkpoint in sorted(root.rglob('model.pt')):
        if 'bert_attention' not in checkpoint.parts:
            continue
        labels_path = checkpoint.parent / 'labels.json'
        labels = json.loads(labels_path.read_text())
        params = signature(checkpoint)
        same = params == final_signature
        run = checkpoint.parent.parent.parent
        if run.name == 'output':
            prepared = run / 'prepared'
        else:
            prepared = run / 'prepared'
        overlap = {}
        if (prepared / 'train.csv').is_file():
            historical = read_csv(prepared / 'train.csv')
            for split in ('val','test'):
                overlap[split] = {}
                for field in ('edition_id','work_id'):
                    old_ids = {r.get(field,'') for r in historical} - {''}
                    current_ids = {r.get(field,'') for r in actual[split]} - {''}
                    overlap[split][field] = len(old_ids & current_ids)
        reports.append(dict(checkpoint=str(checkpoint.relative_to(pipeline)), labels=len(labels),
                            same_learned_block_as_final=same, learned_parameters=params,
                            neighboring_train_file=str((prepared/'train.csv').relative_to(pipeline)),
                            overlap_neighboring_historical_train_with_current=overlap))
    out = dict(final_checkpoint=str(final_checkpoint.relative_to(pipeline)),
               final_learned_parameters=final_signature, checkpoints=reports,
               caution='Neighboring CSVs do not establish checkpoint training membership without a training manifest. Different parameters do not prove a new training split.')
    args.output.write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps([{k:v for k,v in r.items() if k!='learned_parameters'} for r in reports],ensure_ascii=False,indent=2))


if __name__ == '__main__':
    main()
