"""Avalia previsões preservadas: bootstrap por obra, pares, profundidade e erros.

Não treina nem altera modelos. O baseline constante é escolhido apenas no treino.
"""
import argparse
import csv
import json
import math
import random
import statistics
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path
from auditar_titulos_distribuicao import read_csv, write_csv, canonical_code, metrics, sha256

MODELS = {'RL':'logreg', 'F':'F', 'G':'G', 'E':'E'}


def quantile(values, q):
    values = sorted(values)
    position = (len(values)-1)*q
    low = int(position)
    return values[low] + (values[min(low+1,len(values)-1)]-values[low])*(position-low)


def levels(code):
    code=canonical_code(code)
    base, dot, decimal=code.partition('.')
    return [base[:1],base[:2],base,base+'.'+decimal[:1] if decimal else base,code]


def depth(gold, predicted):
    left,right=levels(gold),levels(predicted)
    for i in range(4,-1,-1):
        if left[i]==right[i]:
            return 4-i
    return 5


def category(gold,pred):
    gold,pred=canonical_code(gold),canonical_code(pred)
    if gold.startswith(pred): return 'previsão menos específica'
    if pred.startswith(gold): return 'previsão mais específica'
    if gold[:3]==pred[:3]: return 'código distinto na mesma seção'
    return 'divergência de seção'


def holm(values):
    order=sorted(range(len(values)),key=lambda i:values[i])
    adjusted=[0.0]*len(values)
    previous=0.0
    for j,i in enumerate(order):
        previous=max(previous,min(1.0,values[i]*(len(values)-j)))
        adjusted[i]=previous
    return adjusted


def mcnemar_exact(b,c):
    n=b+c
    return min(1.0,2*sum(math.comb(n,i) for i in range(min(b,c)+1))/(2**n)) if n else 1.0


def load_predictions(path,gold_rows):
    rows=read_csv(path)
    by_id={r['edition_id']:r for r in rows}
    assert len(by_id)==len(rows)==len(gold_rows)
    assert set(by_id)=={r['edition_id'] for r in gold_rows}
    result=[]
    for row in gold_rows:
        pred=by_id[row['edition_id']]
        assert canonical_code(row['cdd'])==canonical_code(pred['cdd_true'])
        result.append(pred)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir',type=Path,required=True)
    parser.add_argument('--llm-run',type=Path,required=True)
    parser.add_argument('--llm-validation-run',type=Path)
    parser.add_argument('--llm-validation-file',type=Path)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--replicates',type=int,default=5000)
    parser.add_argument('--seed',type=int,default=20261122)
    args=parser.parse_args()
    out=args.output_dir
    out.mkdir(parents=True,exist_ok=True)
    splits={s:read_csv(args.run_dir/'prepared'/(s+'.csv')) for s in ['train','val','test']}
    assert [len(splits[s]) for s in ['train','val','test']]==[2423,803,808]
    labels=sorted({canonical_code(r['cdd']) for r in splits['test']})
    frequencies=Counter(canonical_code(r['cdd']) for r in splits['train'])
    majority=sorted(frequencies,key=lambda c:(-frequencies[c],c))[0]
    predictions={}
    table=[]
    for name,stem in MODELS.items():
        predictions[name]=load_predictions(args.run_dir/'results'/('exp_'+stem+'_test.csv'),splits['test'])
        val=load_predictions(args.run_dir/'results'/('exp_'+stem+'_val.csv'),splits['val'])
        test_metrics=metrics(predictions[name],labels)
        report=json.loads((args.run_dir/'results'/('avaliacao_'+stem+'_test.json')).read_text())
        assert test_metrics['hits']==report['accuracy_por_nivel']['full']['hits']
        assert round(test_metrics['macro_f1_labels_present'],4)==report['macro_f1']
        table.append(dict(model=name,validation_accuracy=metrics(val,labels)['accuracy'],**test_metrics))
    predictions['L0']=load_predictions(args.llm_run/'results/exp_A_test_subjects.csv',splits['test'])
    if not args.llm_validation_file and not args.llm_validation_run:
        parser.error('Informe --llm-validation-run ou --llm-validation-file')
    llm_val_path=args.llm_validation_file or args.llm_validation_run/'results/exp_A_val_subjects.csv'
    val=load_predictions(llm_val_path,splits['val'])
    table.append(dict(model='L0',validation_accuracy=metrics(val,labels)['accuracy'],**metrics(predictions['L0'],labels)))
    predictions['MAJ']=[dict(edition_id=r['edition_id'],cdd_true=r['cdd'],cdd_pred=majority) for r in splits['test']]
    val=[dict(cdd_true=r['cdd'],cdd_pred=majority) for r in splits['val']]
    table.append(dict(model='MAJ',validation_accuracy=metrics(val,labels)['accuracy'],**metrics(predictions['MAJ'],labels)))

    groups=defaultdict(list)
    for i,row in enumerate(splits['test']): groups[row.get('work_id') or row['edition_id']].append(i)
    group_indices=list(groups.values())
    totals=[len(g) for g in group_indices]
    hit_by_group={name:[sum(canonical_code(rows[i]['cdd_true'])==canonical_code(rows[i]['cdd_pred']) for i in group) for group in group_indices] for name,rows in predictions.items()}
    rng=random.Random(args.seed)
    samples={name:[] for name in predictions}
    for _ in range(args.replicates):
        picked=rng.choices(range(len(totals)),k=len(totals))
        denominator=sum(totals[i] for i in picked)
        for name in predictions:
            samples[name].append(sum(hit_by_group[name][i] for i in picked)/denominator)
    for row in table:
        row['accuracy_ci95_low']=quantile(samples[row['model']],0.025)
        row['accuracy_ci95_high']=quantile(samples[row['model']],0.975)
    fields=['model','validation_accuracy','n','hits','accuracy','accuracy_ci95_low','accuracy_ci95_high','macro_f1_labels_present','macro_f1_fixed_43','labels_present_n']
    write_csv(out/'metricas_validacao_teste_bootstrap.csv',[{k:r[k] for k in fields} for r in table],fields)

    pair_rows=[]
    names=['L0','E','F','G','RL']
    for a,b in combinations(names,2):
        ca=[canonical_code(r['cdd_true'])==canonical_code(r['cdd_pred']) for r in predictions[a]]
        cb=[canonical_code(r['cdd_true'])==canonical_code(r['cdd_pred']) for r in predictions[b]]
        counts=Counter(zip(ca,cb))
        only_a,only_b=counts[True,False],counts[False,True]
        pair_rows.append(dict(A=a,B=b,only_A=only_a,only_B=only_b,both_correct=counts[True,True],both_wrong=counts[False,False],T_min=min(only_a,only_b),p_exact=mcnemar_exact(only_a,only_b)))
    for row,p in zip(pair_rows,holm([r['p_exact'] for r in pair_rows])):row['p_holm']=p
    write_csv(out/'mcnemar_10_pares.csv',pair_rows,list(pair_rows[0]))

    profiles={}
    hierarchy=[]
    for name in names:
        ds=[depth(r['cdd_true'],r['cdd_pred']) for r in predictions[name]]
        profiles[name]=ds
        counts=Counter(ds)
        hierarchy.append(dict(model=name,**{'depth_'+str(i):counts[i] for i in range(6)},**{'hits_level_'+str(i):sum(levels(r['cdd_true'])[i]==levels(r['cdd_pred'])[i] for r in predictions[name]) for i in range(5)}))
    write_csv(out/'profundidade_e_niveis.csv',hierarchy,list(hierarchy[0]))
    # Matriz de covariância marginal; categorias vazias são retiradas do teste.
    try:
        import numpy as np
        from scipy.stats import chi2
        stuart=[]
        for a,b in combinations(names,2):
            matrix=np.zeros((5,5),dtype=float)
            for left,right in zip(profiles[a],profiles[b]):
                assert left<5 and right<5
                matrix[left,right]+=1
            difference=matrix.sum(axis=1)-matrix.sum(axis=0)
            covariance=-(matrix+matrix.T)
            for i in range(5):covariance[i,i]=matrix[i].sum()+matrix[:,i].sum()-2*matrix[i,i]
            covariance=covariance[:-1,:-1]
            difference=difference[:-1]
            rank=int(np.linalg.matrix_rank(covariance))
            statistic=float(difference@np.linalg.pinv(covariance)@difference)
            stuart.append(dict(A=a,B=b,chi2=statistic,df=rank,p=float(chi2.sf(statistic,rank))))
        for row,p in zip(stuart,holm([r['p'] for r in stuart])):row['p_holm']=p
        write_csv(out/'stuart_maxwell_10_pares.csv',stuart,list(stuart[0]))
    except ImportError:
        stuart=None

    sections=['001','004','005','006','025']
    for name in ['RL','F','G']:
        matrix=Counter((canonical_code(r['cdd_true'])[:3],canonical_code(r['cdd_pred'])[:3]) for r in predictions[name])
        rows=[dict(reference=s,**{p:matrix[s,p] for p in sections}) for s in sections]
        assert sum(sum(r[p] for p in sections) for r in rows)==808
        write_csv(out/('confusao_secoes_'+name+'.csv'),rows,['reference']+sections)

    reference={r['edition_id']:r for r in splits['test']}
    sample=[]
    for model_idx,name in enumerate(['RL','F','G']):
        buckets=defaultdict(list)
        for r in predictions[name]:
            if canonical_code(r['cdd_true'])!=canonical_code(r['cdd_pred']):buckets[canonical_code(r['cdd_true'])[:3]].append(r)
        rng=random.Random(args.seed+model_idx)
        for section in sorted(buckets):
            buckets[section].sort(key=lambda r:r['edition_id'])
            rng.shuffle(buckets[section])
        taken=0
        while taken<30:
            for section in sorted(buckets):
                if buckets[section] and taken<30:
                    r=buckets[section].pop(0)
                    source=reference[r['edition_id']]
                    sample.append(dict(model=name,edition_id=r['edition_id'],titulo=source['titulo'],cdd_true=r['cdd_true'],cdd_pred=r['cdd_pred'],category=category(r['cdd_true'],r['cdd_pred']),texto=source['texto'],status='relação formal automática; interpretação temática e confirmação humana pendentes'))
                    taken+=1
    assert len(sample)==90
    write_csv(out/'amostra_90_erros.csv',sample,list(sample[0]))

    conflicts=[]
    conflicts_summary={}
    for split,rows in list(splits.items())+[('all',[r for rows in splits.values() for r in rows])]:
        by_text=defaultdict(list)
        for r in rows:by_text[r['texto'].strip()].append(r)
        bad=[g for g in by_text.values() if len({canonical_code(r['cdd']) for r in g})>1]
        conflicts_summary[split]=dict(n=len(rows),conflicting_groups=len(bad),involved_records=sum(map(len,bad)),involved_record_rate=sum(map(len,bad))/len(rows))
        for group_idx,group in enumerate(bad):
            for r in group:conflicts.append(dict(scope=split,group=group_idx,edition_id=r['edition_id'],work_id=r.get('work_id',''),titulo=r['titulo'],cdd=r['cdd'],texto=r['texto']))
    if conflicts:write_csv(out/'textos_identicos_referencias_distintas.csv',conflicts,list(conflicts[0]))
    lengths={s:dict(min=min(len(r['texto'].split()) for r in rows),median=statistics.median(len(r['texto'].split()) for r in rows),mean=statistics.mean(len(r['texto'].split()) for r in rows),max=max(len(r['texto'].split()) for r in rows)) for s,rows in splits.items()}
    summary=dict(seed=args.seed,bootstrap=dict(replicates=args.replicates,unit='work_id, substituído por edition_id quando vazio',groups=len(groups),interval='percentil 2,5%–97,5%, mesmos sorteios entre modelos',caution='condicional a modelos fixos; não inclui variabilidade de treinamento nem exclui dependência entre IDs diferentes'),majority_class_chosen_on_training=majority,majority_train_count=frequencies[majority],metrics=table,mcnemar=pair_rows,stuart_maxwell=stuart,depth_profiles=hierarchy,conflicting_text_references=conflicts_summary,text_lengths_whitespace_words=lengths,sample_error_counts={name:dict(Counter(r['category'] for r in sample if r['model']==name)) for name in ['RL','F','G']},sample_unique_editions=len({r['edition_id'] for r in sample}),split_sha256={s:sha256(args.run_dir/'prepared'/(s+'.csv')) for s in splits})
    (out/'auditoria_resultados.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:summary[k] for k in ['majority_class_chosen_on_training','metrics','conflicting_text_references','sample_error_counts','sample_unique_editions','text_lengths_whitespace_words']},ensure_ascii=False,indent=2))


if __name__=='__main__':main()
