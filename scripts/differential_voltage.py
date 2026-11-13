"""Exploratory discharge differential voltage analysis from Silver."""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter, find_peaks, peak_widths
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parent.parent
OUT=ROOT/'outputs/differential_voltage/revised'
SELECT=[2,10,25,50,75,100]
WINDOWS=[11,21,31]

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    records=[]; audit=[]; traces=[]; duplicates=[]; physical=[]
    for path in sorted((ROOT/'outputs/silver').glob('*_silver.parquet')):
        ds=path.stem.split('cycles')[1].split('_',1)[1].replace('_silver','')
        data=pd.read_parquet(path)
        curves={}
        for cycle in sorted(set(n+d for n in SELECT for d in [-1,0,1] if 2<=n+d<=100)):
            g=data[(data.Cycle_Index==cycle)&(data.Phase=='DISCHARGE')]
            if g.Phase_ID.nunique()!=1: raise ValueError('Expected one discharge phase')
            t=g['Test_Time(s)'].to_numpy(float); i=g['Current(A)'].to_numpy(float); v=g['Voltage(V)'].to_numpy(float)
            dt=np.diff(t)
            if not (np.isfinite(t).all() and np.isfinite(i).all() and np.isfinite(v).all() and (dt>=0).all() and (dt<=5).all()): raise ValueError('Invalid discharge interval')
            q=np.r_[0,np.cumsum(-(i[1:]+i[:-1])*.5*dt/3600)]
            grouped=pd.DataFrame(dict(q=q,v=v,t=t)).groupby('q',sort=True).agg(v=('v','median'),lo=('v','min'),hi=('v','max'),count=('v','size'),tmin=('t','min'),tmax=('t','max'))
            bad=grouped.index[(grouped.hi-grouped.lo)>.005].to_numpy()
            for coord,row in grouped[grouped['count']>1].iterrows():
                duplicates.append(dict(cell=ds,cycle=cycle,q_Ah=coord,voltage_spread_V=row.hi-row.lo,time_span_s=row.tmax-row.tmin,count=row['count'],flag=row.hi-row.lo>.005))
            uq=grouped.index.to_numpy(); vv=grouped.v.to_numpy()
            grid=np.linspace(0,uq[-1],301); voltage=np.interp(grid,uq,vv)
            # A common 0.005 Ah grid and fixed physical windows, with a sampling check.
            pg=np.arange(0,uq[-1]+1e-12,.005)
            supported=np.max(np.diff(uq))<=.005
            for pw in WINDOWS:
                pdv=np.gradient(savgol_filter(np.interp(pg,uq,vv),pw,3),pg)
                physical.extend(dict(cell=ds,cycle=cycle,window=pw,span_Ah=(pw-1)*.005,q_Ah=x,q_fraction=x/uq[-1],dV_dQ=y,sampling_supported=supported) for x,y in zip(pg,pdv))
            curves[cycle]=(grid,voltage,{})
            audit.append(dict(cell=ds,cycle=cycle,capacity_Ah=q[-1],samples=len(g),zero_dt=int((dt==0).sum())))
            for w in WINDOWS:
                smooth=savgol_filter(voltage,w,3)
                dv=np.gradient(smooth,grid)
                curves[cycle][2][w]=dv
                interior=-dv[31:270]
                threshold=max(.1,(np.percentile(interior,95)-np.percentile(interior,5))*.05)
                peaks,props=find_peaks(interior,prominence=threshold,distance=15)
                widths=peak_widths(interior,peaks,rel_height=.5)[0] if len(peaks) else []
                peaks=peaks+31
                for p,width in zip(peaks,widths):
                    if p<31 or p>269: continue
                    affected=bool(np.any(abs(bad-grid[p])<=(w//2+1)*(grid[1]-grid[0])))
                    records.append(dict(cell=ds,cycle=cycle,window=w,q_Ah=grid[p],q_fraction=grid[p]/grid[-1],dV_dQ=dv[p],width_Ah=width*(grid[1]-grid[0]),duplicate_affected=affected))
            if cycle in SELECT:
                trace=pd.DataFrame(dict(Q_Ah=q,Voltage_V=v)); trace.to_csv(OUT/f'trace_{ds}_{cycle}.csv',index=False)
                fig,ax=plt.subplots(); ax.plot(q,v); ax.set(xlabel='Discharged capacity (Ah)',ylabel='Voltage (V)',title=f'{ds}, cycle {cycle}'); fig.tight_layout(); fig.savefig(OUT/f'trace_{ds}_{cycle}.png'); plt.close(fig)
        fig,axes=plt.subplots(2,2,figsize=(12,8),constrained_layout=True)
        for n in SELECT:
            q,v,der=curves[n]
            axes[0,0].plot(q,v,label=str(n)); axes[0,1].plot(q/q[-1],v)
            axes[1,0].plot(q,der[21],label=str(n)); axes[1,1].plot(q/q[-1],der[21])
        for ax in axes[0]: ax.set_ylabel('Voltage (V)')
        for ax in axes[1]: ax.set_ylabel('dV/dQ (V/Ah)')
        for ax in axes[:,0]: ax.set_xlabel('Discharged capacity (Ah)')
        for ax in axes[:,1]: ax.set_xlabel('Discharged fraction Q/Q_end')
        axes[0,0].legend(title='Cycle'); fig.suptitle(ds); fig.savefig(OUT/f'overlays_{ds}.png',dpi=160); plt.close(fig)
        fig,axes=plt.subplots(2,3,figsize=(13,7),constrained_layout=True)
        for ax,n in zip(axes.flat,SELECT):
            q,v,der=curves[n]
            for w in WINDOWS: ax.plot(q/q[-1],der[w],label=f'window {w}')
            for neighbor in [n-1,n+1]:
                if neighbor in curves:
                    nq,_,nd=curves[neighbor]; ax.plot(nq/nq[-1],nd[21],color='grey',alpha=.4,ls='--')
            ax.set(title=f'Cycle {n}',xlabel='Q/Q_end',ylabel='dV/dQ (V/Ah)')
        axes.flat[0].legend(); fig.savefig(OUT/f'sensitivity_{ds}.png',dpi=160); plt.close(fig)
        fig,axes=plt.subplots(1,2,figsize=(12,5),constrained_layout=True)
        for n in SELECT:
            q,_,der=curves[n]; axes[0].plot(q[31:270]/q[-1],der[21][31:270],label=str(n))
            pp=pd.DataFrame(physical); pp=pp[(pp.cell==ds)&(pp.cycle==n)&(pp.window==21)&pp.q_fraction.between(.1,.9)]
            axes[1].plot(pp.q_fraction,pp.dV_dQ,label=str(n))
        for ax in axes: ax.set(xlabel='Q/Q_end (interior only)',ylabel='dV/dQ (V/Ah)'); ax.legend(title='Cycle')
        axes[0].set_title('Normalized grid, window 21'); axes[1].set_title('Fixed 0.005 Ah grid, 0.100 Ah window span')
        fig.savefig(OUT/f'interior_comparison_{ds}.png',dpi=160); plt.close(fig)
    peaks=pd.DataFrame(records)
    if not peaks.empty:
        peaks['stable']=False
        for idx,r in peaks.iterrows():
            same=peaks[(peaks.cell==r.cell)&(peaks.cycle==r.cycle)&((peaks.q_fraction-r.q_fraction).abs()<.05)]
            adjacent=peaks[(peaks.cell==r.cell)&((peaks.cycle-r.cycle).abs()==1)&(peaks.window==21)&((peaks.q_fraction-r.q_fraction).abs()<.05)]
            peaks.loc[idx,'stable']=same.window.nunique()==3 and len(adjacent)>0 and not r.duplicate_affected and not same.duplicate_affected.any()
    peaks.to_csv(OUT/'peak_candidates.csv',index=False)
    peaks[(peaks.window==21)&peaks.cycle.isin(SELECT)].to_csv(OUT/'representative_features.csv',index=False)
    pd.DataFrame(duplicates).to_csv(OUT/'duplicate_capacity_audit.csv',index=False)
    pd.DataFrame(physical).to_csv(OUT/'physical_grid_sensitivity.csv',index=False)
    pd.DataFrame(audit).to_csv(OUT/'integration_audit.csv',index=False)
    (OUT/'README.md').write_text('# Differential voltage analysis\n\n- Source: Silver discharge phases; representative cycles 2, 10, 25, 50, 75, 100. Adjacent cycles support sensitivity checks.\n- Q = cumulative trapezoidal integral of -I dt / 3600. No integration across phases. Repeated times add zero capacity; duplicate capacity coordinates use the first voltage.\n- Voltage is interpolated to 301 evenly spaced capacity points. Cubic Savitzky–Golay windows 11, 21 and 31 are compared; numerical gradient gives dV/dQ. Window 21 is the overlay default.\n- Negative derivative means falling voltage. Peaks are local maxima of -dV/dQ; edge regions are excluded. Width is half-prominence width, not electrode capacity.\n- Peak candidates are stable only when matched within 5% discharged fraction in all three windows and at least one available adjacent cycle. This is a heuristic, not statistical confidence or electrode identification.\n- Normalized x axes use Q/Q_end; derivative units remain V/Ah. Grid interpolation does not add measurement resolution.\n- The shrinking capacity span and derivative changes describe apparent loaded-voltage behaviour. Polarization and changing accessible capacity confound electrode interpretation. LLI and LAM cannot be quantified from these high-current curves alone.\n- Files: twelve trace CSVs/PNGs, cell overlay PNGs, sensitivity PNGs, peak_candidates.csv and integration_audit.csv.\n',encoding='utf-8')
    report=OUT/'README.md'
    content=report.read_text(encoding='utf-8').replace('duplicate capacity coordinates use the first voltage','duplicate capacity coordinates use median voltage; spreads above 5 mV are flagged in duplicate_capacity_audit.csv (a provisional review threshold)')
    content+='\n## Revised checks\n\n- Full diagnostic overlays are retained. Interior comparison plots show only 10–90% discharged capacity.\n- Detection and half-prominence widths are computed within the interior; thresholds use its 5th–95th percentile range.\n- Fixed-grid comparison uses 0.005 Ah spacing and windows spanning 0.050, 0.100 and 0.150 Ah. Sampling support is recorded in physical_grid_sensitivity.csv; interpolation adds no resolution.\n- Duplicate-affected features cannot receive the stability flag. Median aggregation remains exploratory where voltage spread is significant; original trace data are retained.\n- representative_features.csv reports position, signed amplitude, width and repeatability. Cross-cycle feature identities and shifts are left unassigned: proximity alone does not establish the same electrode feature.\n- Stability refers only to normalized-window and adjacent-cycle checks, not physical-grid validation or confirmed degradation.\n'
    report.write_text(content,encoding='utf-8')
    print('Created',OUT,'; peak candidates:',len(peaks))

if __name__=='__main__': main()
