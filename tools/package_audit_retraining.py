"""Empacota apenas metadados e resultados auditáveis, sem checkpoints ou caches."""
import argparse
import csv
import hashlib
import io
import json
from pathlib import Path


def digest(data):return hashlib.sha256(data).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--training-root',type=Path,required=True)
    p.add_argument('--audit-dir',type=Path,required=True)
    p.add_argument('--destination',type=Path,required=True)
    args=p.parse_args()
    root,audit,dest=args.training_root.resolve(),args.audit_dir.resolve(),args.destination.resolve()
    replacements=[(str(root),'audit-training'),(str(audit),'audit-workspace'),(str(Path(__file__).resolve().parents[2]),'repository')]
    def clean(value):
        if isinstance(value,dict):return {k:('models/bert_attention' if k=='xlmr_checkpoint' else clean(v)) for k,v in value.items()}
        if isinstance(value,list):return [clean(v) for v in value]
        if isinstance(value,str):
            for old,new in replacements:value=value.replace(old,new)
        return value
    selected=[]
    def add(path,name,mode='copy'):
        assert path.is_file(),path
        selected.append((path,name,mode))
    for seed in [42,43,44]:
        src=root/('seed_'+str(seed)); prefix=src.name+'/'
        manifest=json.loads((src/'training_manifest.json').read_text())
        assert manifest['status']=='complete'
        for split in ['train','val','test']:add(src/'prepared'/(split+'.csv'),prefix+'prepared/'+split+'.csv')
        add(src/'training_manifest.json',prefix+'training_manifest.json','json')
        add(src/'models/bert_attention/labels.json',prefix+'models/bert_attention/labels.json','json')
        for model in ['E','F','G','logreg']:
            for split in ['val','test']:
                for stem,suffix in [('exp_', '.csv'),('avaliacao_', '.json')]:
                    name=stem+model+'_'+split+suffix
                    add(src/'results'/name,prefix+'results/'+name,'json' if suffix=='.json' else 'copy')
        for name in ['selecao_E_val.csv','selecao_G_val.csv','selecao_logreg_val.csv','selecao_final.json','escores_G_val.npz','escores_G_test.npz']:
            add(src/'results'/name,prefix+'results/'+name,'json' if name.endswith('.json') else 'copy')
        for name in ['training.log','selection.log','test.log']:add(src/'logs'/name,prefix+'logs/'+name,'text')
    # Referência LLM preservada: não é treinamento/inferência da rodada nova.
    for name in ['exp_A_test_subjects.csv','avaliacao_A_test_subjects.json']:
        add(root/'seed_42/results'/name,'seed_42/results/'+name,'llm' if name.endswith('.csv') else 'json')
    generic=['auditar_titulos_distribuicao.py','auditar_resultados.py','conferir_nova_rodada.py','nova_rodada_conferencia.json','conferir_checkpoints.py','checkpoints_e_splits.json','treinar_rodada_auditavel.py','proveniencia_coleta.md','proveniencia_coleta.json','conferir_proveniencia_coleta.py','gerar_fluxo_auditavel.py','gerar_matriz_confusao.py','fluxo-metodologia-auditavel.png','fluxo-metodologia-auditavel.svg','matriz-confusao-secoes-RL.png','matriz-confusao-secoes-RL.svg']
    for name in generic:add(audit/name,'audit/'+name,'json' if name.endswith('.json') else 'text' if name.endswith(('.md','.py')) else 'copy')
    for folder in ['resultados-auditaveis','titulos-rodada-auditavel']:
        for path in sorted((audit/folder).iterdir()):
            assert path.suffix in ['.csv','.md','.json']
            add(path,'audit/'+folder+'/'+path.name,'json' if path.suffix=='.json' else 'text' if path.suffix=='.md' else 'copy')
    add(Path(__file__).with_name('audit_retraining_README.md'),'README.md','copy')
    allowed={name for _,name,_ in selected}|{'manifest.json'}
    if dest.exists():
        extra={str(q.relative_to(dest)) for q in dest.rglob('*') if q.is_file()}-allowed
        assert not extra,extra
    records=[]
    for path,name,mode in selected:
        original=path.read_bytes(); data=original
        if mode=='json':data=(json.dumps(clean(json.loads(original)),ensure_ascii=False,indent=2)+'\n').encode()
        elif mode=='text':
            value=original.decode()
            for old,new in replacements:value=value.replace(old,new)
            data=value.encode()
        elif mode=='llm':
            reader=csv.DictReader(io.StringIO(original.decode(),newline=''))
            fields=[x for x in reader.fieldnames if x in {'titulo','source','edition_id','work_id','idioma','cdd_true','cdd_pred','method'}]
            result=io.StringIO(newline=''); writer=csv.DictWriter(result,fieldnames=fields,lineterminator='\n');writer.writeheader()
            writer.writerows({k:r[k] for k in fields} for r in reader);data=result.getvalue().encode()
        target=dest/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(data)
        records.append(dict(path=name,size_bytes=len(data),sha256=digest(data),source_sha256=digest(original),transformation=mode))
    m=dict(schema_version=1,run='2026-11-22-audit-retraining',source_code_commit='564ef1aaaf9ec72f15797459a29f4d3b8d6c160b',primary_seed=42,replication_seeds=[43,44],limitations=['Teste já examinado anteriormente; não é teste externo independente.','Artefatos LLM são referência histórica.','Checkpoints, dumps e respostas brutas não são distribuídos.'],file_count=len(records),total_bytes=sum(r['size_bytes'] for r in records),files=records)
    (dest/'manifest.json').write_text(json.dumps(m,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:m[k] for k in ['file_count','total_bytes','primary_seed','replication_seeds']}))


if __name__=='__main__':main()
