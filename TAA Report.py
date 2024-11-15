# %%
# from dash import Dash, html, dcc, Input, Output
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
import xlwings as xl
import datetime
import TAA_results_dataclass
from sklearn import preprocessing
from matplotlib.font_manager import FontProperties
from matplotlib.backends.backend_pdf import PdfPages
from dataclasses import dataclass
import os,sys
sys.path.insert(1,os.getcwd())
# test comment

# %%
print(os.getcwd())
taa_average = pd.read_excel('taa_performance.xlsx',index_col=0, sheet_name='TAA')*100
taa_macro = pd.read_excel('taa_performance.xlsx',index_col=0, sheet_name='MACRO').fillna(0)
taa_technicals = pd.read_excel('taa_performance.xlsx',index_col=0, sheet_name='TECHNICALS').fillna(0)
taa_combo = pd.read_excel('taa_performance.xlsx',index_col=0, sheet_name='Risk Combo').fillna(0)
prices = pd.read_excel('taa_performance.xlsx',index_col=0, sheet_name='Prices').fillna(0)
bmk_prices = pd.read_excel('taa_performance.xlsx',index_col=0, sheet_name='BMK').fillna(0)
from matplotlib.ticker import FormatStrFormatter
# %%
class color:
   PURPLE = '\033[95m'
   CYAN = '\033[96m'
   DARKCYAN = '\033[36m'
   BLUE = '\033[94m'
   GREEN = '\033[92m'
   YELLOW = '\033[93m'
   RED = '\033[91m'
   BOLD = '\033[1m'
   UNDERLINE = '\033[4m'
   END = '\033[0m'

date =  datetime.datetime.now().strftime("%m/%d/%Y")
title = f'Current TAA (as of {date})'


# %%

fig, axs = plt.subplots(sharey=True)


bp1 = axs.boxplot(taa_average[:-1].values)
# plt.plot(np.arange(1,8),[taa_average.iloc[-1].values],'ro')
plt.plot(1,taa_average.iloc[-1,0],'bo',label='Curr. TAA Weights')
plt.plot(2,taa_average.iloc[-1,1],'bo')
plt.plot(3,taa_average.iloc[-1,2],'bo')
plt.plot(4,taa_average.iloc[-1,3],'bo')
plt.plot(5,taa_average.iloc[-1,4],'bo')
plt.plot(6,taa_average.iloc[-1,5],'bo')
plt.plot(7,taa_average.iloc[-1,6],'bo')
bl = [ 'EQ', 'GOV', 'IG', 'HY', 'COMM', 'GOLD', 'CASH']
axs.set_xticklabels(bl, fontsize = 10)
axs.set_title('Current TAA Summary')
axs.yaxis.set_major_formatter(FormatStrFormatter('%.1f%%'))
# axs.axhline(linewidth=0.5, linestyle="--")
axs.grid(axis='y')
axs.legend()
plt.savefig('current_taa_bp.png')

# Boxplot points for further analysis
fliers=bp1["fliers"]
caps=bp1["caps"]
boxes=bp1["boxes"]

asset_dict = {'EQ':0, 'GOV':1, 'IG':2, 'HY':3, 'COMM':4, 'GOLD':5, 'CASH':6}
def get_bp_data(bp1: dict, asset: int):
    fliers=bp1["fliers"]
    caps=bp1["caps"]
    boxes=bp1["boxes"]
    outliers=fliers[asset]
    # lo_thresh = caps[]

# %%
eq_outliers = fliers[0].get_data()
lo_eq_thresh = caps[0].get_data()
hi_eq_thresh = caps[1].get_data()
eq_outliers_pts = taa_average[taa_average.iloc[:,0].eq(eq_outliers[1][0])]
eq_outlier = eq_outliers[1].shape

hy_outliers = fliers[3].get_data()
lo_hy_thresh = caps[6].get_data()
hi_hy_thresh = caps[7].get_data()
hy_outliers_pts = taa_average[taa_average.iloc[:,3].eq(hy_outliers[1][0])]
hy_outlier = hy_outliers[1].shape
for i in range(1,hy_outlier[0]):
    x = taa_average[taa_average.iloc[:,3].eq(hy_outliers[1][i])]
    hy_outliers_pts = pd.concat([hy_outliers_pts,x])

gold_outliers = fliers[5].get_data()
lo_gold_thresh = caps[10].get_data()
hi_gold_thresh = caps[11].get_data()
gold_outlier = gold_outliers[1].shape
gold_outliers_pts = taa_average[taa_average.iloc[:,5].eq(gold_outliers[1][0])]
for i in range(1,gold_outlier[0]):
    x = taa_average[taa_average.iloc[:,5].eq(gold_outliers[1][i])]
    gold_outliers_pts = pd.concat([gold_outliers_pts,x])

cash_outliers = fliers[-1].get_data()
lo_cash_thresh = caps[-2].get_data()
hi_cash_thresh = caps[-1].get_data()
cash_outlier = cash_outliers[1].shape
cash_outliers_pts = taa_average[taa_average.iloc[:,-1].eq(cash_outliers[1][0])]
for i in range(1,cash_outlier[0]):
    x = taa_average[taa_average.iloc[:,-1].eq(cash_outliers[1][i])]
    cash_outliers_pts = pd.concat([cash_outliers_pts,x])

# %%
fig = plt.figure()
gs = fig.add_gridspec(2,2)
# (option for gridspec) wspace=0, hspace=0
axs = gs.subplots(sharex=True, sharey=True)
# (option for subplots) sharex=True, sharey=True
fig.suptitle('Outliers (since 2004)', fontsize=14, fontweight='bold')

axs[0,0].plot(taa_average.iloc[:,0], label='TAA Equity Weight')
axs[0,0].plot(eq_outliers_pts['EQ'], 'ko', label='Equity Outlier')
axs[0,0].axhline(lo_eq_thresh[1][0],color='r')
axs[0,0].axhline(hi_eq_thresh[1][0],color='r')
axs[0,0].grid()
# EQ pt for last update
# axs[0,0].annotate(f'{hi_eq_thresh[1][0]}',(taa_average.index[-1],-0.04))
axs[0,0].set_title(f'EQ ({eq_outlier[0]} pt(s))', fontsize=8)
# axs[0,0].set_ylim([-0.2,0.2])
axs[0,0].yaxis.set_major_formatter(FormatStrFormatter('%.1f%%'))

axs[0,1].plot(taa_average.iloc[:,3], label='TAA HY Weight')
axs[0,1].plot(hy_outliers_pts['HY'], 'ko', label=f'HY Outlier')
axs[0,1].axhline(lo_hy_thresh[1][0],color='r')
axs[0,1].axhline(hi_hy_thresh[1][0],color='r')
axs[0,1].set_title(f'HY ({hy_outlier[0]} pt(s))', fontsize=8)
axs[0,1].grid()

axs[1,0].plot(taa_average.iloc[:,5], label='TAA Gold Weight')
axs[1,0].plot(gold_outliers_pts['GOLD'], 'ko', label=f'GOLD Outlier')
axs[1,0].axhline(lo_gold_thresh[1][0],color='r')
axs[1,0].axhline(hi_gold_thresh[1][0],color='r')
axs[1,0].set_title(f'GOLD ({hy_outlier[0]} pt(s))', fontsize=8)
axs[1,0].grid()

axs[1,1].plot(taa_average.iloc[:,-1], label='TAA Cash Weight')
axs[1,1].plot(cash_outliers_pts['CASH'], 'ko', label=f'Cash Outlier')
axs[1,1].axhline(lo_cash_thresh[1][0],color='r')
axs[1,1].axhline(hi_cash_thresh[1][0],color='r')
axs[1,1].set_title(f'Cash ({cash_outlier[0]} pt(s))', fontsize=8)
axs[1,1].grid()

fig.savefig('outliers.png')

# %%
data_show = pd.concat([taa_average.iloc[-1],
                       taa_average.iloc[-5],
                       taa_average.iloc[-10]], axis=1)
data_show.columns = data_show.columns.strftime("%m/%d/%Y")
data_show_t = data_show.transpose()
blue = np.array(plt.cm.Blues(0.5))
orange = np.array(plt.cm.Oranges(0.5))
green = np.array(plt.cm.Greens(0.5))
fig, ax = plt.subplots()
n = np.arange(7)
w = 0.2
offset = 0.2
data_ = data_show_t.values
x1=ax.bar(n-offset,data_[0,:],width=w, label=taa_average.index[-1].strftime("%m/%d/%Y"), color = blue)
x2=ax.bar(n,data_[1,:],width=w, label=taa_average.index[-5].strftime("%m/%d/%Y"), color= orange)
x3=ax.bar(n+offset,data_[2,:],width=w, label=taa_average.index[-10].strftime("%m/%d/%Y"), color= green)
x4=ax.axhline(y=0,color='k')
bl = ['', 'EQ', 'GOV', 'IG', 'HY', 'COMM', 'GOLD', 'CASH']
ax.set_title('Current vs Past TAA', fontsize = 12)
ax.grid(True, linewidth=0.5)

ax.set_xticklabels(bl, fontsize = 10)
yticks = np.arange(-10.0,8.0,2.0)
ax.set_yticklabels(yticks, fontsize = 10)
ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f%%'))
# ylabels = [f'\\{x:%.1f%%}' for x in yticks]
plt.rcParams['font.size'] = 8
plt.xticks([])
ax.set_ylim([-12,7])
ax.set_ylabel('Active Weight (%)')
# ax.bar_label(x1,padding=3, fmt=lambda x: np.where(abs(x) > 0, f'{x:.2f}', ''))
# ax.xticks(n, data_show_t.columns)
perc = data_show_t.values/100
perc_fmt = np.array([[f"{i:.1%}" for i in val] for val in perc])

column_headers = ['EQ', 'GOV', 'IG', 'HY', 'COMM', 'GOLD', 'CASH']
# plt.cm.BuPu(np.full(len(column_headers), 0.1))

the_table = plt.table(cellText=perc_fmt,
                      rowLabels=data_show.transpose().index,
                      colLabels=data_show.transpose().columns,
                      colColours=plt.cm.BuPu(np.full(len(column_headers), 0.2)),
                      cellColours=np.stack([[blue,blue,blue,blue,blue,blue,blue],
                                            [orange,orange,orange,orange,orange,orange,orange],
                                            [green,green,green,green,green,green,green]]),
                      rowColours=np.stack([blue,orange,green]),
                      loc='bottom')
plt.legend()
for (row,col), cell in the_table.get_celld().items():
    if row==0:
        cell.set_text_props(fontproperties=FontProperties(weight='bold'))

plt.savefig('taa_bars_past3.png', bbox_inches='tight')


# %%
# Current TAA decomposition
fig,ax = plt.subplots()
plt.title(title, fontsize=12)
salmon = plt.cm.OrRd(0.3)
turq = plt.cm.YlGnBu(0.5)
gold = plt.cm.cividis(0.8)
blue = plt.cm.Blues(0.8)
plt.plot(taa_average.iloc[-1].values*100,'yd', label = 'TAA', markersize=6, color=blue)
plt.axhline(y=0, color='k')

plt.bar(taa_macro.columns,taa_macro.iloc[-1].values*100, width=0.3, color=salmon, label = 'Macro')
plt.bar(taa_technicals.columns,taa_technicals.iloc[-1].values*100, width=0.3, color=turq, label = 'Technicals')
plt.bar(taa_combo.columns,taa_combo.iloc[-1].values*100, width=0.3, color=gold, label = 'Risk Combo')
plt.grid(True)
ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f%%'))
ax.set_ylabel('Active Weight (%)')
plt.xticks([])

data_show = pd.concat([taa_macro.iloc[-1],taa_technicals.iloc[-1],taa_combo.iloc[-1], taa_average.iloc[-1]], axis=1)
data_show_t_v = data_show.transpose().values

data_show_fmt = np.array([[f"{i:.1%}" for i in val] for val in data_show_t_v])

curr_table = plt.table(cellText=data_show_fmt,
                       rowLabels=['MACRO', 'TECHNICAL', 'COMBO', 'AVERAGE'],
                       colLabels=data_show.transpose().columns,
                       colColours=plt.cm.BuPu(np.full(len(column_headers), 0.2)),
                       cellColours=np.stack([[salmon,salmon,salmon,salmon,salmon,salmon,salmon],
                                             [turq,turq,turq,turq,turq,turq,turq],
                                             [gold,gold,gold,gold,gold,gold,gold],
                                             [blue,blue,blue,blue,blue,blue,blue]]),
                       rowColours=np.stack([salmon,turq,gold,blue]),
                       loc='bottom')

for (row,col), cell in curr_table.get_celld().items():
    if row==0:
        cell.set_text_props(fontproperties=FontProperties(weight='bold'))

# plt.legend(loc = 'lower center')
plt.legend()
plt.savefig('taa_current_decomp.png', bbox_inches='tight')

# %%
price_pct = prices.pct_change()
taa_ret = (taa_average.iloc[:-1].values@price_pct.iloc[:-1].transpose().values)
taa_ret = np.diag(taa_ret[:,1:])+1
taa_ret[0] = taa_ret[0]*100
taa_ret = pd.DataFrame(data=taa_ret, index=taa_average.index[2:])
plt.plot(taa_ret.cumprod())
plt.title('TAA Cumulative Performance')
plt.savefig('taa_cum_perf.png')

# %%
taa_combo_decomp = pd.read_excel('taa_performance.xlsx',index_col=0, sheet_name='Risk Combo Comp').fillna(0)
fig, ax = plt.subplots()
prd = 12
offset = 0.2
n = np.arange(prd)
taa_combo_decomp_prd = taa_combo_decomp.iloc[-prd:]*100
plt.bar(n-offset,taa_combo_decomp_prd['MoMo'], width=offset, label='MoMo')
plt.bar(n,taa_combo_decomp_prd['CAST'], width=offset, label='CAST')
plt.bar(n+offset,taa_combo_decomp_prd['FCI'], width=offset, label='FCI')
plt.plot(taa_combo_decomp_prd['COMBO'].values,'ko-', markersize=6, label = 'Combo')
plt.xticks(n, taa_combo_decomp_prd.index.strftime("%m/%d/%Y"), rotation="vertical")
plt.legend()
ax.yaxis.set_major_formatter(FormatStrFormatter('%.0f%%'))
ax.set_ylabel('Risk OFF Prob.(%)')
plt.rcParams['font.size'] = 10
plt.rcParams
plt.title('Risk Combo Components')
plt.savefig('risk_combo_details.png', bbox_inches = 'tight')

# %%
# Technical
fig, ax = plt.subplots()
prd = 4
n = np.arange(prd)
w = 0.1
offset = 0.1
data_tech_prd = taa_technicals.iloc[-prd:]
x1=ax.bar(n-3*offset,data_tech_prd['EQ']*100,width=w, label='EQ')
x2=ax.bar(n-2*offset,data_tech_prd['GOV']*100,width=w, label='GOV')
x3=ax.bar(n-1*offset,data_tech_prd['IG']*100,width=w, label='IG')
x4=ax.bar(n,data_tech_prd['HY']*100,width=w, label='HY')
x5=ax.bar(n+1*offset,data_tech_prd['COMM']*100,width=w, label='COMM')
x6=ax.bar(n+2*offset,data_tech_prd['CASH']*100,width=w, label='CASH')
ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f%%'))
ax.set_title('Techincals', fontsize = 12)
ax.grid(True, linewidth=0.5, axis='y', which='both')
ax.set_ylabel('Active Weight(%)')
ax.set_xticks([0, 1, 2, 3])
ax.set_xticklabels(taa_technicals.index[-prd:].strftime("%m/%d/%Y"), fontsize = 10)
# ylabels = [f'\\{x:%.1f%%}' for x in yticks]
plt.rcParams['font.size'] = 8

# ax.bar_label(x1,padding=3, fmt=lambda x: np.where(abs(x) > 0, f'{x:.2f}', ''))

fig.legend(loc='lower left', bbox_to_anchor=(0.2, 0.2, 0.5, 0.5,),frameon=True)
fig.savefig('taa_technicals.png')

# %%
combo_thresholds = np.array([[0.025, 0.05, -0.05, 0.1, -0.125,	-0.025, 0.025],
                             [-0.025, -0.05,	0.05,	-0.1,	0.125, 0.025, -0.025]])
taa_combo_ret = (taa_combo.iloc[:-1].values@price_pct.iloc[:-1].transpose().values)
taa_combo_ret = np.diag(taa_combo_ret[:,1:])+1
taa_combo_ret[0] = taa_combo_ret[0]*100
taa_combo_ret = pd.DataFrame(data=taa_combo_ret, index=taa_average.index[2:])
plt.plot(taa_combo_ret.cumprod())
plt.title('Risk Combo Cumulative Performance')
plt.savefig('combo_perf.png')

# %%
taa_tech_ret = (taa_technicals.iloc[:-1].values@price_pct.iloc[:-1].transpose().values)
taa_tech_ret = np.diag(taa_tech_ret[:,1:])+1
taa_tech_ret[0] = taa_tech_ret[0]*100
taa_tech_ret = pd.DataFrame(data=taa_tech_ret, index=taa_average.index[2:])
plt.plot(taa_tech_ret.cumprod())
plt.title('Technicals Cumulative Performance')
plt.savefig('tech_perf.png')


# %%
taa_macro_ret = (taa_macro.iloc[:-1].values@price_pct.iloc[:-1].transpose().values)
taa_macro_ret = np.diag(taa_macro_ret[:,1:])+1
taa_macro_ret[0] = taa_macro_ret[0]*100
taa_macro_ret = pd.DataFrame(data=taa_macro_ret, index=taa_average.index[2:])
plt.plot(taa_macro_ret.cumprod())
plt.title('Econ Backdrop Cumulative Performance')
plt.savefig('macro_perf.png')
# %%
# bmk_pct = price_pct[['EQ','GOV','CASH']]
# taa_bmk_ret = (np.array([.6,.4,-1])@bmk_pct[1:-1].transpose().values)+1
taa_bmk_ret = (np.array([1/6,1/6,1/6,1/6,1/6,1/6,-1])@price_pct.iloc[1:-1].transpose().values)+1
taa_bmk_ret[0] = taa_bmk_ret[0]*100
taa_bmk_ret = pd.DataFrame(data=taa_bmk_ret, index=taa_average.index[2:])

# %%
bmk_pct = bmk_prices.pct_change()
bmk_w = np.array([9,20,5,3,8,5,8,10,4,3,3,2,5,5,2,3,2,1,2])/100
bmk_ret = (bmk_w@bmk_pct.iloc[1:-1].transpose().values)+1
bmk_ret[0] = bmk_ret[0]*100
bmk_ret = pd.DataFrame(data=bmk_ret, index=taa_average.index[2:])

eq_bmk = bmk_w[0:5]/sum(bmk_w[0:5])
gov_bmk = bmk_w[6:12]/sum(bmk_w[6:12])
ig_bmk = bmk_w[12:14]/sum(bmk_w[12:14])
hy_bmk = bmk_w[14:16]/sum(bmk_w[14:16])
comm_bmk = bmk_w[16:18]/sum(bmk_w[16:18])
# gold_bmk = bmk_w[18]
# cash_bmk = bmk_w[5]
fig, ax = plt.subplots()
# all_bmk = np.concatenate([eq_bmk,gov_bmk,ig_bmk,hy_bmk,comm_bmk,bmk_w[[18,5]]])
r,c = bmk_pct.shape
active_pos = np.zeros(bmk_pct.shape)
active_combo_pos = np.zeros(bmk_pct.shape)
active_econ_pos = np.zeros(bmk_pct.shape)
active_tech_pos = np.zeros(bmk_pct.shape)
taa_average /= 100
for i in range(r):
    active_pos[i,0:5] = taa_average.iloc[i,0]*eq_bmk
    active_pos[i,6:12] = taa_average.iloc[i,1]*gov_bmk
    active_pos[i,12:14] = taa_average.iloc[i,2]*ig_bmk
    active_pos[i,14:16] = taa_average.iloc[i,3]*hy_bmk
    active_pos[i,16:18] = taa_average.iloc[i,4]*comm_bmk
    active_pos[i,18] = taa_average.iloc[i,5]
    active_pos[i,5] = taa_average.iloc[i,6]
    ###
    active_econ_pos[i,0:5] = taa_macro.iloc[i,0]*eq_bmk
    active_econ_pos[i,6:12] = taa_macro.iloc[i,1]*gov_bmk
    active_econ_pos[i,12:14] = taa_macro.iloc[i,2]*ig_bmk
    active_econ_pos[i,14:16] = taa_macro.iloc[i,3]*hy_bmk
    active_econ_pos[i,16:18] = taa_macro.iloc[i,4]*comm_bmk
    active_econ_pos[i,18] = taa_macro.iloc[i,5]
    active_econ_pos[i,5] = taa_macro.iloc[i,6]
    ###
    active_combo_pos[i,0:5] = taa_combo.iloc[i,0]*eq_bmk
    active_combo_pos[i,6:12] = taa_combo.iloc[i,1]*gov_bmk
    active_combo_pos[i,12:14] = taa_combo.iloc[i,2]*ig_bmk
    active_combo_pos[i,14:16] = taa_combo.iloc[i,3]*hy_bmk
    active_combo_pos[i,16:18] = taa_combo.iloc[i,4]*comm_bmk
    active_combo_pos[i,18] = taa_combo.iloc[i,5]
    active_combo_pos[i,5] = taa_combo.iloc[i,6]
    ###
    active_tech_pos[i,0:5] = taa_technicals.iloc[i,0]*eq_bmk
    active_tech_pos[i,6:12] = taa_technicals.iloc[i,1]*gov_bmk
    active_tech_pos[i,12:14] = taa_technicals.iloc[i,2]*ig_bmk
    active_tech_pos[i,14:16] = taa_technicals.iloc[i,3]*hy_bmk
    active_tech_pos[i,16:18] = taa_technicals.iloc[i,4]*comm_bmk
    active_tech_pos[i,18] = taa_technicals.iloc[i,5]
    active_tech_pos[i,5] = taa_technicals.iloc[i,6]

active_pos2 = active_pos + bmk_w
active_ret = active_pos2[:-1]@bmk_pct.transpose().values
active_ret = np.diag(active_ret[:,1:])+1
active_ret[0] = 100
active_ret_pd = pd.DataFrame(data=active_ret, index=taa_average.index[1:])

active_econ_pos2 = active_econ_pos + bmk_w
active_econ_ret = active_econ_pos2[:-1]@bmk_pct.transpose().values
active_econ_ret = np.diag(active_econ_ret[:,1:])+1
active_econ_ret[0] = 100
active_econ_ret_pd = pd.DataFrame(data=active_econ_ret, index=taa_macro.index[1:])

active_combo_pos2 = active_combo_pos + bmk_w
active_combo_ret = active_combo_pos2[:-1]@bmk_pct.transpose().values
active_combo_ret = np.diag(active_combo_ret[:,1:])+1
active_combo_ret[0] = 100
active_combo_ret_pd = pd.DataFrame(data=active_combo_ret, index=taa_combo.index[1:])

active_tech_pos2 = active_tech_pos + bmk_w
active_tech_ret = active_tech_pos2[:-1]@bmk_pct.transpose().values
active_tech_ret = np.diag(active_tech_ret[:,1:])+1
active_tech_ret[0] = 100
active_tech_ret_pd = pd.DataFrame(data=active_tech_ret, index=taa_technicals.index[1:])

plt.plot(active_ret_pd.cumprod(), 'k', label='TAA')
plt.plot(bmk_ret.cumprod(), 'x--',label='ABS Benchmark', markersize=2)
plt.plot(taa_bmk_ret.cumprod(), '--',label='1/N')
plt.title('TAA vs ABS Benchmark vs 1/N')

ax.set_ylabel('Price')

plt.legend()
plt.savefig('all_perf.png')
def yearly_perfomrance(years: list, ptf: pd.DataFrame):
    lbl = 0
    for i in range(len(years)):
        plt.boxplot(ptf.loc[years[i]], labels=lbl)
        lbl += 1
    plt.savefig('yearly_perf.png')

# %%
fig, ax = plt.subplots()
years=['2020','2021','2022' ]
perf_data2005 = active_ret_pd.loc['2005'].values-1
perf_data2006 = active_ret_pd.loc['2006'].values-1
perf_data2007 = active_ret_pd.loc['2007'].values-1
perf_data2008 = active_ret_pd.loc['2008'].values-1
perf_data2009 = active_ret_pd.loc['2009'].values-1
perf_data2010 = active_ret_pd.loc['2010'].values-1
perf_data2011 = active_ret_pd.loc['2011'].values-1
perf_data2012 = active_ret_pd.loc['2012'].values-1
perf_data2013 = active_ret_pd.loc['2013'].values-1
perf_data2014 = active_ret_pd.loc['2014'].values-1
perf_data2015 = active_ret_pd.loc['2015'].values-1
perf_data2016 = active_ret_pd.loc['2016'].values-1
perf_data2017 = active_ret_pd.loc['2017'].values-1
perf_data2018 = active_ret_pd.loc['2018'].values-1
perf_data2019 = active_ret_pd.loc['2019'].values-1
perf_data2020 = active_ret_pd.loc['2020'].values-1
perf_data2021 = active_ret_pd.loc['2021'].values-1
perf_data2022 = active_ret_pd.loc['2022'].values-1
perf_data2023 = active_ret_pd.loc['2023'].values-1
perf_data2024 = active_ret_pd.loc['2024']

perf_macro2019 = active_econ_ret_pd.loc['2019'].values-1
perf_macro2020 = active_econ_ret_pd.loc['2020'].values-1
perf_macro2021 = active_econ_ret_pd.loc['2021'].values-1
perf_macro2022 = active_econ_ret_pd.loc['2022'].values-1
perf_macro2023 = active_econ_ret_pd.loc['2023'].values-1
perf_macro2024 = active_econ_ret_pd.loc['2024']

perf_combo2019 = active_combo_ret_pd.loc['2019'].values-1
perf_combo2020 = active_combo_ret_pd.loc['2020'].values-1
perf_combo2021 = active_combo_ret_pd.loc['2021'].values-1
perf_combo2022 = active_combo_ret_pd.loc['2022'].values-1
perf_combo2023 = active_combo_ret_pd.loc['2023'].values-1
perf_combo2024 = active_combo_ret_pd.loc['2024']

perf_tech2019 = active_tech_ret_pd.loc['2019'].values-1
perf_tech2020 = active_tech_ret_pd.loc['2020'].values-1
perf_tech2021 = active_tech_ret_pd.loc['2021'].values-1
perf_tech2022 = active_tech_ret_pd.loc['2022'].values-1
perf_tech2023 = active_tech_ret_pd.loc['2023'].values-1
perf_tech2024 = active_tech_ret_pd.loc['2024']


# perf_data = np.concatenate([perf_data2005,perf_data2006,perf_data2007,perf_data2008,perf_data2009,perf_data2010,perf_data2011,perf_data2012,perf_data2013,perf_data2014,perf_data2015,perf_data2016,perf_data2017,perf_data2018,perf_data2019,perf_data2020,perf_data2021,perf_data2022,perf_data2023], axis=1)
perf_data = np.concatenate([perf_data2019,perf_data2020,perf_data2021[:-1],perf_data2022,perf_data2023], axis=1)*100
years=['2019','2020','2021', '2022','2023']
perf_data_pd = pd.DataFrame(data=perf_data, columns=years)
plt.boxplot(perf_data_pd,labels=years)
ax.axhline(linewidth=0.5, linestyle="--")
ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f%%'))
ax.grid(axis='y')
ax.set_ylabel('Active Weight (%)')
plt.title('Returns Distribution (2019-2023)')
plt.savefig('ret_dist2019-2023.png')
# %%
fig, ax = plt.subplots()
plt.boxplot((perf_data2024-1)*100)
plt.plot(1,(perf_data2024.iloc[-1].values-1)*100,'bo',label=perf_data2024.index[-1].strftime("%m/%d/%Y"))
plt.plot(1,(perf_data2024.iloc[-5].values-1)*100,'kx',label=perf_data2024.index[-5].strftime("%m/%d/%Y"))
plt.plot(1,(perf_data2024.iloc[-9].values-1)*100,'gv',label=perf_data2024.index[-9].strftime("%m/%d/%Y"))
plt.xlabel('2024 YTD')
plt.title('Returns TAA YTD (2024)')
plt.legend()
ax.axhline(linewidth=0.5, linestyle="--")
ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f%%'))
ax.grid(axis='y')
ax.set_ylabel('Active Returns (%)')
plt.savefig('ret_dist2024.png')

# %%

returns = {'2019': np.squeeze(perf_data2019), '2020': np.squeeze(perf_data2020), '2021': np.squeeze(perf_data2021), '2022': np.squeeze(perf_data2022), '2023': np.squeeze(perf_data2023), '2024': np.squeeze(perf_data2024.values-1)}
# returns = {'2019': np.squeeze(perf_data2019).tolist(), '2020': np.squeeze(perf_data2020).tolist(), '2024': np.squeeze(perf_data2024.values-1).tolist()}
# returns = {'ABC': [34.54, 34.345, 34.761], 'DEF': [34.541, 34.748, 34.482]}
fig, ax = plt.subplots()
plt.boxplot(returns.values())
ax.axhline(linewidth=0.5, linestyle="--")
ax.grid(axis='y')
plt.title('Return Distributions (2019-2024)')
plt.xticks([1,2,3,4,5,6],list(returns.keys()))
plt.savefig('all_rets.png')
# %%
fig, ax = plt.subplots()
plt.plot((perf_data2024-1)*100)
plt.xticks(rotation="vertical")
ax.axhline(linewidth=0.5, linestyle="--")
ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f%%'))

ax.grid(axis='y')

# %%
def returns_dist(data: pd.DataFrame, title: str, nm_png: str):
    fig, ax = plt.subplots()
    years=['2019','2020','2021', '2022','2023']
    data_pd = pd.DataFrame(data=data, columns=years)
    plt.boxplot(data_pd, labels=years)
    plt.title(title)
    ax.axhline(linewidth=0.5, linestyle="--")
    ax.grid(axis='y')
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f%%'))
    ax.set_ylabel('Active Returns (%)')
    plt.savefig(nm_png)


returns_dist(data=np.concatenate([perf_data2019*100,perf_data2020*100,perf_data2021[:-1]*100,perf_data2022*100,perf_data2023*100], axis=1),
             title='Returns Distribution (2019-2023)',
             nm_png='taa_ret_box.png')
# %% 
returns_dist(data=np.concatenate([perf_macro2019*100,perf_macro2020*100,perf_macro2021[:-1]*100,perf_macro2022*100,perf_macro2023*100], axis=1),
             title='Returns Distribution Econ Backdrop (2019-2023)',
             nm_png='taa_macro_ret_box.png') 
# %%
returns_dist(data=np.concatenate([perf_combo2019*100,perf_combo2020*100,perf_macro2021[:-1]*100,perf_combo2022*100,perf_combo2023*100], axis=1),
             title='Returns Distribution Combo (2019-2023)',
             nm_png='taa_combo_ret_box.png')

# %%
returns_dist(data=np.concatenate([perf_tech2019*100,perf_tech2020*100,perf_macro2021[:-1]*100,perf_tech2022*100,perf_tech2023*100], axis=1),
             title='Returns Distribution Technicals (2019-2023)',
             nm_png='taa_tech_ret_box.png')
# %%
track_err_ABS = active_ret_pd.iloc[2:] - bmk_ret.iloc[1:]
track_err_1N = active_ret_pd.iloc[2:] - taa_bmk_ret.iloc[1:]

r,c = track_err_ABS.shape
info_ratio_ABS = np.zeros((r,1))
info_ratio_1N = np.zeros((r,1))
hit_ratio_ABS = np.zeros((r,1))
hit_ratio_1N = np.zeros((r,1))
for i in range(1,r):
    info_ratio_ABS[i] = track_err_ABS.iloc[i].values / track_err_ABS.iloc[:i].std()
    info_ratio_1N[i] = track_err_1N.iloc[i].values / track_err_1N.iloc[:i].std()
    

r2,c2 = info_ratio_ABS.shape
info_ratio_ABS_pd = pd.DataFrame(data=info_ratio_ABS[2:], index=track_err_ABS.index[2:])
info_ratio_1N_pd = pd.DataFrame(data=info_ratio_1N[2:], index=track_err_ABS.index[2:])

cur_ABS_hit = track_err_ABS[track_err_ABS>0].dropna().shape
cur_1N_hit = track_err_1N[track_err_1N>0].dropna().shape
print(f'Hit Ratio (vs ABS): {format(cur_ABS_hit[0]/r2,".2%")}')
print(f'Hit Ratio (vs 1/N): {format(cur_1N_hit[0]/r2,".2%")}')

# %%
fig, ax = plt.subplots(1,3)
ax[0] = plt.boxplot(active_ret_pd[1:],showmeans=True)
ax[0] = plt.title('TAA Active Return')

ax[1] = plt.boxplot(bmk_ret[1:],showmeans=True)
ax[1] = plt.title('ABS Benchmark Return')

ax[2] = plt.boxplot(taa_bmk_ret[1:],showmeans=True)
ax[2] = plt.title('1/N Benchmark Return')

# plt.plot(bmk_ret.cumprod(),'r',label='TAA Benchmark', markersize=1, lw=2)
# %%
# plt.plot(taa_macro_ret.cumprod(),label='Econ Backdrop')
# plt.plot(taa_tech_ret.cumprod(),label='Technicals')
# plt.plot(taa_combo_ret.cumprod(),label='Risk Combo')
# plt.plot(taa_bmk_ret.cumprod(),'r',label='TAA Overall', markersize=1, lw=2)
# plt.plot(taa_ret.cumprod(),'ko-',label='TAA Overall', markersize=1, lw=2)
# plt.title('Cumulative Performances')
# plt.legend()
# plt.savefig('all_perf.png')
with open('test.txt','rb') as rd:
    txt = rd.read().decode('latin-1')
# %%
import fpdf
width = 210
height = 297
report = fpdf.FPDF()

# Overall
report.add_page()
report.set_font('Arial', 'B', 20)
report.cell(40,10,f'TAA Report')
report.image('taa_current_decomp.png', 5, 30, 150,100)
report.image('taa_bars_past3.png', 5, 150, 150, 100)

# Overall - distribution
report.add_page()
report.set_font('Arial', 'B', 20)
report.cell(0,10,f'TAA Report: Distribution', ln=True)
report.image('current_taa_bp.png', 5, 30, 120, 80)
report.image('box_plot_expl.png', 120, 30, 70, 60)
report.set_font('Arial', '', 10)
# report.multi_cell(200,5, txt)
report.ln()
# report.cell(w=50,h=210,txt='Ttestdfa',border=1, align='L')

report.image('outliers.png', 5,130, 160, 120)

# report.image('equity_outliers.png', 5, 100, width-120, 80)
# report.image('hy_outliers.png', 100, 100, width-120, 80)
# report.image('gold_outliers.png', 5, 180, width-120, 80)
# report.image('cash_outliers.png', 100, 180, width-120, 80)

# Risk Combo details
report.add_page()
report.set_font('Arial', 'B', 16)
report.cell(40,10,f'Risk Combo: Past 12 weeks')
report.image('risk_combo_details.png', 5, 30, width-10)



# Technicals details
report.add_page()
report.set_font('Arial', 'B', 16)
report.cell(40,10,f'Technicals: Past 4 weeks')
report.image('taa_technicals.png', 5, 30, width-10)

# Performance
report.add_page()
report.set_font('Arial', 'B', 16)
report.cell(40,10,f'Perfomance: Yearly')
report.image('all_perf.png', 5, 30, width-10)
report.image('ret_dist2019-2023.png', 5, 170, 120, 100)
report.image('ret_dist2024.png', 120, 170, 80, 100)

report.add_page()
report.set_font('Arial', 'B', 16)
report.cell(40,10,f'Perfomance (2019-2023): by Component')
report.image('taa_ret_box.png', 5, 30, 110, 80)
report.image('taa_macro_ret_box.png', 100, 30, 110, 80)
report.image('taa_combo_ret_box.png', 5, 130, 110, 80)
report.image('taa_tech_ret_box.png', 100, 130, 110, 80)


# appendix
report.add_page()
report.set_font('Arial', 'B', 18)
report.cell(40,10,f'Appendix: Benchmark')
report.image('abs_bmk.png', 5, 30, width-10)
report.image('bmk_details.png', 5, 140, w=width-100, h=100)
report.image('active_current.png', 120, 140)

# report.image('combo_perf.png', 5, 150, width-10)

report.output('taa_report.pdf')
# %%
from fpdf import FPDF

# create FPDF object
# Layout ('P','L')
# Unit ('mm', 'cm', 'in')
# format ('A3', 'A4' (default), 'A5', 'Letter', 'Legal', (100,150))
pdf = FPDF('P', 'mm', 'Letter')

# Add a page
pdf.add_page()

# specify font
# fonts ('times', 'courier', 'helvetica', 'symbol', 'zpfdingbats')
# 'B' (bold), 'U' (underline), 'I' (italics), '' (regular), combination (i.e., ('BU'))
pdf.set_font('helvetica', 'BIU', 16)
pdf.set_text_color(220,50,50)
# Add text
# w = width
# h = height
# txt = your text
# ln (0 False; 1 True - move cursor down to next line)
# border (0 False; 1 True - add border around cell)
pdf.cell(120, 100, 'Hello World!', ln=True, border=False)

pdf.set_font('times', '', 12)
pdf.cell(80, 10, 'Good Bye World!',border=1)

pdf.output('pdf_1.pdf')
