"""Post hoc stride/gate diagnosis from retained hourly scores. No API calls.

The computed outcomes are deterministic gate alarms, not live Jev predictions.
"""
from pathlib import Path
import hashlib
import json
import numpy as np
from hydrojev.benchmarks.hydrojev_paper_metrics import hold_grid, hourly_metrics

ROOT=Path(__file__).resolve().parents[2]
SOURCE=ROOT/'artifacts/hydrojev_detection_v2/paper_summary.json'
CACHE=ROOT/'artifacts/hydrojev_detection_v2/summary_derivations/matched_evidence_scores.npz'
MANIFEST=CACHE.with_name('matched_evidence_manifest.json')
OUT=ROOT/'artifacts/hydrojev_sampling_diagnostic'

def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    summary=json.loads(SOURCE.read_text(encoding='utf-8'))
    manifest=json.loads(MANIFEST.read_text(encoding='utf-8'))
    assert digest(CACHE)==manifest['score_file_sha256']
    data=np.load(CACHE,allow_pickle=False);rows=[];events=[]
    for ds in summary['variants']:
        labels=data[ds+'__raw_labels'];start=int(data[ds+'__start'][0])
        hourly=np.array([np.isfinite(data[ds+'__'+mid]) & (data[ds+'__'+mid]>=summary['fit']['thresholds'][name]) for name,mid in manifest['method_ids'].items()])
        for window in [6,1]:
            windows=np.stack([hourly[:,max(0,j-window+1):j+1].any(axis=1) for j in range(hourly.shape[1])],axis=1)
            votes=windows.sum(axis=0)
            for stride in [6,3,2,1]:
                indices=np.arange(start,len(labels),stride)
                for minimum in [3,2,1]:
                    candidate=votes[indices-start]>=minimum
                    m=hourly_metrics(labels,hold_grid(indices,candidate,len(labels)),start)
                    r={'dataset':ds,'stride_h':stride,'window_h':window,'minimum_votes':minimum,
                       'decision_points':len(indices),'candidate_points':int(candidate.sum()),
                       'event_hits':m['detected_attack_scenarios'],'event_total':m['attack_scenarios'],
                       'benign_hour_fpr':m['benign_hour_fpr'],'mean_detected_ttd_h':m['mean_ttd_hours'],
                       'batadal_S':m['batadal_score']['S'],'per_attack':m['per_attack'],
                       'interpretation':'deterministic gate-as-alarm; live Jev not evaluated'}
                    rows.append(r)
                    if window==6 and stride==6 and minimum==3:
                        expected=summary['variants'][ds]['no_jev']['metrics']
                        for field in ['scenario_recall','benign_hour_fpr','mean_ttd_hours']:
                            assert np.isclose(m[field],expected[field],rtol=1e-12,atol=1e-12),field
                        assert np.isclose(r['batadal_S'],expected['batadal_score']['S'],rtol=1e-12,atol=1e-12)
            if window==6:
                for number,e in enumerate(summary['variants'][ds]['full']['metrics']['per_attack'],1):
                    a,b=e['start_index']-start,e['end_index']-start;ev=votes[a:b]
                    events.append({'dataset':ds,'event':number,'start_index':a+start,'end_index_exclusive':b+start,
                        'duration_hours':b-a,'max_alerting_streams':int(ev.max()),
                        'hourly_candidates_by_minimum_votes':{str(k):int(np.sum(ev>=k)) for k in [1,2,3]},
                        'stream_alert_hours':{name:int(windows[i,a:b].sum()) for i,name in enumerate(manifest['method_ids'])}})
    report={'status':'post_hoc_diagnostic','api_calls':0,'live_jev_new_protocol_status':'not_run',
            'notes':['Original benchmark and manuscript unchanged.','These results cannot establish Jev performance on newly eligible states.',
                     'Lowering the gate to two while retaining a two-vote comparator still makes every candidate a control alarm.'],
            'source_hashes':{str(p.relative_to(ROOT)):digest(p) for p in [SOURCE,CACHE,MANIFEST]},'rows':rows,'event_diagnostics':events}
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/'diagnostic.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    selected=[r for r in rows if r['dataset']=='test_dataset' and r['window_h']==6 and r['minimum_votes']==3]
    table='\n'.join(f"| {r['stride_h']} | {r['decision_points']} | {r['candidate_points']} | {r['event_hits']}/{r['event_total']} | {r['benign_hour_fpr']:.2%} | {r['mean_detected_ttd_h']:.2f} |" for r in selected)
    event_table='\n'.join(f"| E{e['event']} | {e['duration_hours']} | {e['max_alerting_streams']} | {e['hourly_candidates_by_minimum_votes']['3']} | {e['hourly_candidates_by_minimum_votes']['2']} |" for e in events if e['dataset']=='test_dataset')
    text=f'''# 判定频率与候选门控诊断

日期：2026-09-24。使用已保存的逐小时分数和训练阈值，未重训、未调用 API、未改原实验与论文。本报告是事后诊断，不是新增 live Jev 成绩。

固定6小时因果证据窗口及至少3个报警检测器的门控，仅改变判定间隔：

| 间隔/h | 判定点 | 候选点 | 候选覆盖事件 | 良性小时误报率 | 条件平均延迟/h |
|---|---:|---:|---:|---:|---:|
{table}

表中报警是确定性候选规则的输出，也是当前门控下 Jev 可能达到的事件覆盖上限；不是 Jev 的新实测结果。加密判定改变报警时刻和持续时间，因此误报率、延迟可以变化。

逐小时、固定6小时窗口的事件内证据：

| 事件 | 持续时间/h | 最大报警检测器数 | 至少3票的小时数 | 至少2票的小时数 |
|---|---:|---:|---:|---:|
{event_table}

本轮状态新鲜且可用，确定性对照在候选内只需至少2个检测器报警，但候选门控先要求至少3个。因此所有候选都已被对照报警。逐点满足：Jev报警集合包含于候选集合，候选集合等于规则对照报警集合。同一网格与前向保持下，Jev无法新增检出事件或更早首报，只能抑制误报、维持报警或弃判。这个约束不能用于判断 Jev 在其他输入和路由下是否有能力提高召回。

“HydroJEV，无 Jev”表示组件替换消融，表格更清楚的名称是“确定性检测器融合对照（Deterministic detector-fusion control）”。该对照不是完整 HydroJEV。

要检验新增检出能力，需先在开发数据上定义更宽的评估状态集合，例如每小时全部状态，让 Jev 与固定规则对照都接触相同输入，并比较相同误报预算下的召回和延迟。仅把门控降为2票而保留2票对照，仍没有给 Jev 新增报警的空间。增加时间点不等于增加独立攻击事件；更多事件和管网需另外获取。

完整敏感性矩阵见 diagnostic.json。不能从该离线矩阵推断新方案的 live Jev 优势。
'''
    (OUT/'DIAGNOSIS.md').write_text(text,encoding='utf-8')
    print(json.dumps({'test_fixed_window':[{k:v for k,v in r.items() if k!='per_attack'} for r in selected],
        'missed_events':[e for e in events if e['dataset']=='test_dataset' and e['hourly_candidates_by_minimum_votes']['3']==0]},ensure_ascii=False,indent=2))

if __name__=='__main__':main()
