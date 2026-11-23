"""Confere escores, partições e logs das três execuções sem nova inferência."""
import argparse
import hashlib
import json
import re
import statistics
from pathlib import Path
import numpy as np
from auditar_titulos_distribuicao import read_csv, canonical_code, metrics


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    reports=[]
    for seed in [42,43,44]:
        run=args.root/('seed_'+str(seed))
        manifest=json.loads((run/'training_manifest.json').read_text())
        assert manifest['status']=='complete'
        for split in ['train','val','test']:
            assert hashlib.sha256((run/'prepared'/(split+'.csv')).read_bytes()).hexdigest()==manifest['split_sha256'][split]
        epochs=re.findall(r'epoch (\d+).*val_acc ([0-9.]+)',(run/'logs/training.log').read_text())
        best=max(epochs,key=lambda r:float(r[1]))
        selected=json.loads((run/'results/selecao_final.json').read_text())
        alpha=selected['ensemble_g']['alpha_xlmr']
        check={}
        scores={}
        for split in ['val','test']:
            gold=read_csv(run/'prepared'/(split+'.csv'))
            f=read_csv(run/'results'/('exp_F_'+split+'.csv'))
            g=read_csv(run/'results'/('exp_G_'+split+'.csv'))
            assert [r['edition_id'] for r in gold]==[r['edition_id'] for r in f]==[r['edition_id'] for r in g]
            with np.load(run/'results'/('escores_G_'+split+'.npz'),allow_pickle=False) as z:
                labels=z['labels'].tolist()
                assert np.allclose(z['xlmr'].sum(axis=1),1,atol=1e-5)
                assert np.allclose(z['knn'].sum(axis=1),1,atol=1e-5)
                pf=[labels[i] for i in z['xlmr'].argmax(axis=1)]
                pg=[labels[i] for i in (alpha*z['xlmr']+(1-alpha)*z['knn']).argmax(axis=1)]
            assert pf==[r['cdd_pred'] for r in f]
            assert pg==[r['cdd_pred'] for r in g]
            check[split]={'n':len(gold),'neural_matches_archived_csv':True,'ensemble_matches_archived_csv':True}
            scores[split]={'F':metrics(f,labels),'G':metrics(g,labels)}
        reports.append(dict(seed=seed,epochs_executed=len(epochs),best_epoch=int(best[0]),alpha_xlmr=alpha,score_consistency=check,metrics=scores,arguments=manifest['arguments'],versions=manifest['versions'],split_sha256=manifest['split_sha256']))
    assert all(r['split_sha256']==reports[0]['split_sha256'] for r in reports)
    aggregate={}
    for m in ['F','G']:
        aggregate[m]={}
        for measure in ['accuracy','macro_f1_labels_present']:
            values=[r['metrics']['test'][m][measure] for r in reports]
            aggregate[m][measure]={'mean':statistics.mean(values),'sample_std':statistics.stdev(values),'values':values}
    result=dict(primary_seed=42,replication_seeds=[43,44],test_seen_previously=True,all_current_splits_identical_across_seeds=True,runs=reports,aggregate=aggregate)
    args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(dict(aggregate=aggregate,runs=[{k:r[k] for k in ['seed','epochs_executed','best_epoch','alpha_xlmr','score_consistency']} for r in reports]),ensure_ascii=False,indent=2))


if __name__=='__main__':main()
