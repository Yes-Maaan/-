# -*- coding: utf-8 -*-
"""
Created on Tue Nov  8 15:22:27 2022

@author: biubiubiu123
"""

import xarray as xr  # 文件读取
import numpy as np  # 数据处理，矩阵
from eofs.standard import Eof  # eof分析模块
import cartopy.crs as ccrs  # 地图
import cartopy.feature as cfeature
import matplotlib.pyplot as plt  # 绘图
import cartopy.mpl.ticker as cticker


# 图片参数设置函数
def contour_map(fig, img_extent, spec):  # fig画布，img_extent地图边界参数，spec调整显示ticks范围
    fig.set_extent(img_extent, crs=ccrs.PlateCarree())
    fig.add_feature(cfeature.COASTLINE.with_scale('50m'))  # 海岸线分辨率50m
    fig.add_feature(cfeature.LAKES, alpha=0.5)  # 加湖泊
    fig.set_xticks(np.arange(img_extent[0], img_extent[1] + spec, spec), crs=ccrs.PlateCarree())
    fig.set_yticks(np.arange(img_extent[2], img_extent[3] + spec, spec), crs=ccrs.PlateCarree())
    lon_formatter = cticker.LongitudeFormatter()
    lat_formatter = cticker.LatitudeFormatter()
    fig.xaxis.set_major_formatter(lon_formatter)
    fig.yaxis.set_major_formatter(lat_formatter)


# 读取文件数据
f = xr.open_dataset("D:/Python_data/out.nc")
dot = np.array(f['cspath'].loc[:, 0:90, :])
lat = f['lat'].loc[0:90]
lon = f['lon']
# years = range(1979, 2018)
# years = f['year']
years = f.coords['year']  # 三种形式的years等价
# 计算EOF
solver = Eof(dot)
eof = solver.eofsAsCorrelation(neofs=2)  # 空间模态，取前2个
pc = solver.pcs(npcs=2, pcscaling=1)  # 时间序列取前两个
var = solver.varianceFraction()  # 计算模态的方差
# 绘图
fig = plt.figure(figsize=(15, 15))  # 画布
proj = ccrs.PlateCarree(central_longitude=80)  # 投影
left, right, lower, upper = (0, 160, 10, 90)  # 地图边界
img_extent = [left, right, lower, upper]
# 空间模态绘制
f_ax1 = fig.add_axes([0.1, 0.3, 0.3, 0.4], projection=proj)  # EOF1
contour_map(f_ax1, img_extent, 20)  # 设置图片各参数
f_ax1.set_title('(a) EOF1', loc='left')  # 左标题
f_ax1.set_title('%.2f%%' % (var[0] * 100), loc='right')  # 右标题，解释方差
f_ax1.contourf(lon, lat, eof[0, :, :], levels=np.arange(-0.9, 1.0, 0.1), zorder=0, extend='both',
               transform=ccrs.PlateCarree(), cmap=plt.cm.RdBu_r)  # 绘制填色
f_ax2 = fig.add_axes([0.1, 0.1, 0.3, 0.4], projection=proj)  # EOF2
contour_map(f_ax2, img_extent, 20)
f_ax2.set_title('(b) EOF2', loc='left')
f_ax2.set_title('%.2f%%' % (var[1] * 100), loc='right')
c2 = f_ax2.contourf(lon, lat, eof[1, :, :], levels=np.arange(-0.9, 1.0, 0.1), zorder=0, extend='both',
                    transform=ccrs.PlateCarree(), cmap=plt.cm.RdBu_r)
# 绘制色标
position = fig.add_axes([0.1, 0.18, 0.3, 0.017])  # colorbar的位置
fig.colorbar(c2, cax=position, orientation='horizontal', format='%.1f')  # colorbar的样式
# 时间序列绘制
f_ax3 = fig.add_axes([0.45, 0.42, 0.3, 0.156])  # 绘制PC1
f_ax3.set_title('(b) PC1', loc='left')
f_ax3.set_ylim(-3, 3)  # y轴上下限
f_ax3.axhline(0, linestyle="-", c='k')  # 水平参考线
f_ax3.plot(years, pc[:, 0], c='k')  # 绘制折线
pc1_9 = np.convolve(pc[:, 0], np.repeat(1 / 9, 9), mode='valid')  # 计算九年滑动平均
f_ax3.plot(years[4:-4], pc1_9, c='r', lw=2)  # 绘制滑动平均
f_ax4 = fig.add_axes([0.45, 0.22, 0.3, 0.156])  # 绘制PC2
f_ax4.set_title('(d) PC2', loc='left')
f_ax4.set_ylim(-3, 3)
f_ax4.axhline(0, linestyle="-", c='k')
f_ax4.plot(years, pc[:, 1], c='k')
pc2_9 = np.convolve(pc[:, 1], np.repeat(1 / 9, 9), mode='valid')
f_ax4.plot(years[4:-4], pc2_9, c='r', lw=2)

plt.savefig("F:/python_test/eof.pdf", dpi=1000, bbox_inches='tight')  # 保存图片，dpi图片分辨率，
plt.show()
