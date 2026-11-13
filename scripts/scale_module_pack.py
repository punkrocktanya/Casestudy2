"""Illustrative capacity-envelope scenarios, NOT a validated electrothermal pack model."""
import json
import hashlib
import numpy as np
import pandas as pd
from individual_curve_rul import INPUT
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT=INPUT.parents[1]/'pack_scaling'
CONFIG=dict(series_groups_per_module=4,parallel_cells=2,modules_in_series=3,
            ensemble_size=100,seed=42,capacity_cv=.03,rate_cv=.15,
            reference_temperature_K=298.15,activation_energy_J_mol=20000,
            maximum_temperature_offset_K=10,initial_soc_spread=.03,
            balance_current_A=.05,balance_hours=1,active_transfer_efficiency=.85,horizon=300)


def envelope(capacities,usable_soc):
    """Shape: ensemble,module,series group,parallel cell. Topology-aware envelope."""
    group=(capacities*usable_soc).sum(axis=-1)
    module=group.min(axis=-1)
    return module,module.min(axis=-1)


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    data=pd.read_csv(INPUT,dtype={'Dataset_ID':str})
    rng=np.random.default_rng(CONFIG['seed']); rows=[]; traces=[]
    fig,axes=plt.subplots(1,2,figsize=(12,4.5),constrained_layout=True)
    for ax,(cell,g) in zip(axes,data.groupby('Dataset_ID')):
        # Only the early exponential coefficient supplies a scenario baseline.
        from fit_capacity_fade import fit
        h=g[g.Cycle_Index<=8].sort_values('Cycle_Index'); q0=h.Discharge_Capacity_Ah.iloc[0]
        k=fit('Exponential',h.Cycle_Index.to_numpy()-2,h.Cycle_Index.to_numpy()-2,h.Discharge_Capacity_Ah.to_numpy())[1]
        shape=(CONFIG['ensemble_size'],CONFIG['modules_in_series'],CONFIG['series_groups_per_module'],CONFIG['parallel_cells'])
        initial=q0*np.exp(rng.normal(-.5*CONFIG['capacity_cv']**2,CONFIG['capacity_cv'],size=shape))
        rates=k*np.exp(rng.normal(-.5*CONFIG['rate_cv']**2,CONFIG['rate_cv'],size=shape))
        # End-of-charge SOC deficit: balancing has finite time/current, not unlimited correction.
        deficit=rng.uniform(0,CONFIG['initial_soc_spread'],size=shape)
        gradient=np.linspace(0,CONFIG['maximum_temperature_offset_K'],np.prod(shape[1:])).reshape((1,)+shape[1:])
        temp=CONFIG['reference_temperature_K']+gradient
        acceleration=np.exp(CONFIG['activation_energy_J_mol']/8.314*(1/CONFIG['reference_temperature_K']-1/temp))
        for label,thermal,balancing in [('uniform_no_balance',False,'none'),('gradient_no_balance',True,'none'),
                                        ('gradient_passive',True,'passive'),('gradient_active',True,'active')]:
            effective_rate=rates*(acceleration if thermal else 1)
            if balancing=='none':soc=1-deficit;loss=np.zeros(shape[0])
            elif balancing=='passive':
                # Bleed excess from high-SOC cells towards the lowest SOC in each series module.
                target=deficit.max(axis=(-2,-1),keepdims=True)
                removed=np.minimum(np.maximum(target-deficit,0)*initial,CONFIG['balance_current_A']*CONFIG['balance_hours'])
                soc=1-deficit-removed/initial;loss=(removed*3.7).sum(axis=(1,2,3))
            else:
                target=deficit.mean(axis=(-2,-1),keepdims=True)
                imbalance=(deficit-target)*initial
                donor=np.maximum(-imbalance,0).sum(axis=(-2,-1),keepdims=True)
                recipient=np.maximum(imbalance,0).sum(axis=(-2,-1),keepdims=True)
                cap=CONFIG['balance_current_A']*CONFIG['balance_hours']
                transferred=np.minimum(np.minimum(donor,recipient/CONFIG['active_transfer_efficiency']),cap)
                removed=np.divide(np.maximum(-imbalance,0)*transferred,donor,out=np.zeros_like(initial),where=donor>0)
                added=np.divide(np.maximum(imbalance,0)*transferred*CONFIG['active_transfer_efficiency'],recipient,out=np.zeros_like(initial),where=recipient>0)
                soc=1-deficit+(added-removed)/initial
                loss=(removed-added).sum(axis=(1,2,3))*3.7
                assert np.all(loss>=-1e-10)
            assert np.all((soc>=0)&(soc<=1+1e-10))
            # Common ideal fully charged BOL reference isolates accessibility losses between strategies.
            _,bol=envelope(initial,np.ones_like(initial))
            eol=np.full(shape[0],np.nan)
            for elapsed in range(CONFIG['horizon']+1):
                capacity=initial*np.exp(-effective_rate*elapsed)
                modules,pack=envelope(capacity,soc)
                ratio=pack/bol
                newly=np.isnan(eol)&(ratio<.8);eol[newly]=elapsed+2
                traces.append(dict(Dataset_ID=cell,Scenario=label,Cycle_Index=elapsed+2,
                                   Mean_Retention=float(ratio.mean()),P05_Retention=float(np.quantile(ratio,.05)),
                                   P95_Retention=float(np.quantile(ratio,.95))))
            rows.extend(dict(Dataset_ID=cell,Scenario=label,Sample=i,Predicted_EOL_Cycle=v,
                             Initial_Balance_Loss_Wh=float(loss[i])) for i,v in enumerate(eol))
            block=pd.DataFrame(traces);block=block[(block.Dataset_ID==cell)&(block.Scenario==label)]
            ax.plot(block.Cycle_Index,block.Mean_Retention,label=label.replace('_',' '))
        ax.axhline(.8,color='black',ls='--');ax.set(title=f'{cell}-based synthetic pack',xlabel='Cycle index',ylabel='Usable Ah / ideal BOL Ah',xlim=(2,50),ylim=(.3,1.05));ax.legend(fontsize=7);ax.grid(alpha=.2)
    fig.savefig(OUT/'scenario_retention.png',dpi=180);plt.close(fig)
    result=pd.DataFrame(rows);result.to_csv(OUT/'scenario_samples.csv',index=False)
    pd.DataFrame(traces).to_csv(OUT/'scenario_trajectories.csv',index=False)
    summary=result.groupby(['Dataset_ID','Scenario']).agg(Mean_EOL=('Predicted_EOL_Cycle','mean'),Min_EOL=('Predicted_EOL_Cycle','min'),Max_EOL=('Predicted_EOL_Cycle','max'),Mean_Initial_Balance_Loss_Wh=('Initial_Balance_Loss_Wh','mean')).reset_index()
    summary.to_csv(OUT/'scenario_summary.csv',index=False)
    (OUT/'assumptions.json').write_text(json.dumps(dict(**CONFIG,source_sha256=hashlib.sha256(INPUT.read_bytes()).hexdigest(),interpretation='Synthetic scenarios only; uncertainty bands reflect assumed parameter distributions, not measured population confidence'),indent=2))
    (OUT/'README.md').write_text('''# Illustrative module/pack capacity envelopes

4S2P modules, three in series (12S2P, 24 cells). Positive capacity and fade-rate spreads are sampled from explicitly assumed lognormal distributions. Early cell exponential fits through cycle 8 supply rate baselines. Assumed 0-10 K gradients modify rates through an uncalibrated Arrhenius factor. These are sensitivity inputs, NOT estimated thermal parameters or manufacturing distributions.

Parallel capacities are summed and series-group capacities limited by the minimum. This is a charge-availability envelope: no voltage/OCV model, unequal parallel current dynamics, thermal ODE, calendar ageing, resistance growth or BMS voltage cutoff is simulated. Actual delivered pack capacity can be lower. Independent cell envelopes do not validate real pack lifetime.

No balancing retains initial SOC deficits. Passive balancing bleeds excess charge toward the lowest SOC, bounded by current and duration; it does not create usable charge. Active redistribution is bounded by donor charge, recipient deficit, current, time and transfer efficiency. Loss uses an illustrative 3.7 V constant. The corrected SOC margins are held fixed over ageing; balance heat and repeated balancing are not fed back into degradation. This isolates initial-accessibility scenarios rather than claiming a full balancing lifetime benefit. All scenarios use the same ideal fully charged BOL denominator.

Report these plots only as synthetic sensitivity demonstrations. Needed for validation: more matched cells, current/voltage traces, temperature/cooling data, interconnection resistances and measured balancer behaviour. Ensemble bands are assumed scenario spread, not calibrated probability intervals. No actual pack EOL is claimed.
''',encoding='utf-8')
    print(summary.to_string(index=False))


if __name__=='__main__':main()
