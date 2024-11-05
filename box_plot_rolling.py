#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 10 13:49:05 2024

@author: Stephen
"""
import openpyxl
import pandas as pd
import numpy as np
from pypfopt import EfficientFrontier
from pypfopt import risk_models
from pypfopt import expected_returns
import math
from scipy.linalg import sqrtm
import seaborn as sns
import matplotlib.pyplot as plt


### upload excel file



##### 25% in domestic, increase in sharpe ratio, full time period (2007-2023), break into 4 groups
sharpe_ratio = pd.read_excel('/Users/Stephen/Desktop/Learning_code/portfolio_diversification/rolling_window/Sharperatio_for_rolling.xlsx', '25%domestic_4section')

df = pd.DataFrame(sharpe_ratio)

plt.figure(figsize=(8,6))

sns.boxplot(x='Scenario', y = 'Sharpe ratio', data =df, color = 'lightgray', fliersize=0)

sns.stripplot(x='Scenario', y="Sharpe ratio", data=df, color = 'black', alpha=0.7, jitter=True)

plt.ylim(-0.6, 0.8)

## show the plot
plt.title('Difference in Sharpe ratio relative to domestic index \n Rolling window optimisation with 25% domestic, 2007-2024')
plt.show()



##### 25% in domestic, difference in sharpe ratio relative to no restriction, full time period (2007-2023), 44 index
sharpe_ratio = pd.read_excel('/Users/Stephen/Desktop/Learning_code/portfolio_diversification/rolling_window/Sharperatio_for_rolling.xlsx', '25%_4section_decrease_2norestri')

df = pd.DataFrame(sharpe_ratio)


plt.figure(figsize=(8,6))

sns.boxplot(x='Scenario', y = 'Sharpe ratio', data =df, color = 'lightgray', fliersize=0)

sns.stripplot(x='Scenario', y="Sharpe ratio", data=df, color = 'black', alpha=0.7, jitter=True)

plt.ylim(-0.6, 0.15)

## show the plot
plt.title('Difference in sharpe ratio relative to no geopolitical restriction \n Rolling window optimisation with 25% domestic, 2007-2024')
plt.ylabel('Sharpe ratio')
plt.show()



##### 25% in domestic, difference in sharpe ratio relative to no restriction, full time period (2007-2023), 44 index
sharpe_ratio = pd.read_excel('/Users/Stephen/Desktop/Learning_code/portfolio_diversification/rolling_window/Sharperatio_for_rolling.xlsx', '25%_4section_mean_diff%')

df = pd.DataFrame(sharpe_ratio)

# Convert Sharpe ratio to percentage
df['Sharpe ratio (%)'] = df['Sharpe ratio'] * 100

# Create a bar chart
fig, ax = plt.subplots()
ax.bar(df['Scenario'], df['Sharpe ratio (%)'], color='skyblue')

# Add labels and title
ax.set_xlabel('Scenario')
ax.set_ylabel('Sharpe Ratio (%)')
ax.set_title('Average % Reduction in sharpe ratio relative to no geopolitical restriction \n Rolling window optimisation with 25% domestic, 2007-2024')

# Show gridlines for better readability
ax.yaxis.grid(True, linestyle='--', alpha=0.7)

plt.ylim(-100, 0)

plt.tight_layout()
plt.show()



##### 25% and 50% in domestic , difference in sharpe ratio relative to no restriction, full time period (2007-2023), geo-friends only vs no restriction 
sharpe_ratio = pd.read_excel('/Users/Stephen/Desktop/Learning_code/portfolio_diversification/rolling_window/Sharperatio_for_rolling.xlsx', '25_50%_4section_decre_2nores')

df = pd.DataFrame(sharpe_ratio)

plt.figure(figsize=(8,6))

sns.boxplot(x='Scenario', y = 'Sharpe ratio', data =df, color = 'lightgray', fliersize=0)

sns.stripplot(x='Scenario', y="Sharpe ratio", data=df, color = 'black', alpha=0.7, jitter=True)

plt.ylim(-0.6, 0.2)

## show the plot
plt.title('Difference in sharpe ratio relative to no geopolitical restriction \n Optimised portfolio with different % of domestic, 2007-2024')
plt.show()




##### Time series  full time period (2007-2023), all 4 groups relative to own idnex
sharpe_ratio = pd.read_excel('/Users/Stephen/Desktop/Learning_code/portfolio_diversification/rolling_window/Sharperatio_for_rolling.xlsx', 'time-changeinsharpe_rel2own')

df = pd.DataFrame(sharpe_ratio)


# Pivot the DataFrame to have time as index and objects as columns
df_pivot = df.pivot(index='Date', columns='Object', values='Value')

# Plot the data
fig, ax = plt.subplots()
df_pivot.plot(ax=ax)

# Add labels and title
ax.set_xlabel('Time')
ax.set_ylabel('Sharpe ratio')
ax.set_title('Difference in Sharpe ratio relative to domestic index \n Rolling window optimisation with 25% domestic, 2007-2024')

ax.legend(title='Scenario', fontsize='small', title_fontsize='small')

plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()





##### Time series  full time period (2007-2023), 3 groups relative to no restriction group
sharpe_ratio = pd.read_excel('/Users/Stephen/Desktop/Learning_code/portfolio_diversification/rolling_window/Sharperatio_for_rolling.xlsx', 'time-Diffinchangeshar_2norest')

df = pd.DataFrame(sharpe_ratio)


# Pivot the DataFrame to have time as index and objects as columns
df_pivot = df.pivot(index='Date', columns='Object', values='Value')

# Plot the data
fig, ax = plt.subplots()
df_pivot.plot(ax=ax)

# Add labels and title
ax.set_xlabel('Time')
ax.set_ylabel('Sharpe ratio')
ax.set_title('Difference in Sharpe ratio relative to no geopolitical restriction\n Rolling window optimisation with 25% domestic, 2007-2024')


ax.legend(title='Scenario', fontsize='small', title_fontsize='small')

plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()





##### 12m rolling sharpe ratio Time series  full time period (2007-2023), all 4 groups relative to own idnex
sharpe_ratio = pd.read_excel('/Users/Stephen/Desktop/Learning_code/portfolio_diversification/rolling_window/Sharperatio_for_rolling.xlsx', '12mtime-changeinsharpe_rel2own')

df = pd.DataFrame(sharpe_ratio)


# Pivot the DataFrame to have time as index and objects as columns
df_pivot = df.pivot(index='Date', columns='Object', values='Value')

# Plot the data
fig, ax = plt.subplots()
df_pivot.plot(ax=ax)

# Add labels and title
ax.set_xlabel('Time')
ax.set_ylabel('Sharpe ratio')
ax.set_title('Difference in 12m rolling Sharpe ratio relative to domestic index \n Rolling window optimisation with 25% domestic, 2007-2024')

ax.legend(title='Scenario', fontsize='small', title_fontsize='small')

plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()




'''
##### 12m rolling sharpe ratio Time series  full time period (2007-2023), 3 groups relative to no restriction group
sharpe_ratio = pd.read_excel('/Users/Stephen/Desktop/Learning_code/portfolio_diversification/rolling_window/Sharperatio_for_rolling.xlsx', '12mtime-Diffinchangeshar_2nore')

df = pd.DataFrame(sharpe_ratio)


# Pivot the DataFrame to have time as index and objects as columns
df_pivot = df.pivot(index='Date', columns='Object', values='Value')

# Plot the data
fig, ax = plt.subplots()
df_pivot.plot(ax=ax)

# Add labels and title
ax.set_xlabel('Time')
ax.set_ylabel('Sharpe ratio')
ax.set_title('Difference in 12m rolling Sharpe ratio relative to no geopolitical restriction\n Rolling window optimisation with 25% domestic, 2007-2024')


ax.legend(title='Scenario', fontsize='small', title_fontsize='small')

plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

'''

##### 12m rolling sharpe ratio Time series  full time period (2007-2023), 3 groups relative to no restriction group
### add VIX


import pandas as pd
import matplotlib.pyplot as plt

# Read the Excel file and create the DataFrame
sharpe_ratio = pd.read_excel('/Users/Stephen/Desktop/Learning_code/portfolio_diversification/rolling_window/Sharperatio_for_rolling.xlsx', '12mtime-Diffinchangeshar_2nore')
df = pd.DataFrame(sharpe_ratio)

# Pivot the DataFrame to have time as index and columns as objects
df_pivot = df.pivot(index='Date', columns='Object', values='Value')

# Debug prints
print("Available columns:", df_pivot.columns.tolist())
print("\nFirst few rows of the data:")
print(df_pivot.head())

# Check VIX data specifically
if 'VIX' in df_pivot.columns:
    print("\nVIX data summary:")
    print(df_pivot['VIX'].describe())
    print("\nFirst few VIX values:")
    print(df_pivot['VIX'].head())
else:
    print("\nVIX column not found in the DataFrame")

# Create the plot
fig, ax1 = plt.subplots(figsize=(12, 6))

# Plot Sharpe ratios (excluding VIX)
for col in df_pivot.columns:
    if col != 'VIX':
        ax1.plot(df_pivot.index, df_pivot[col], label=col)

# Set up primary axis
ax1.set_xlabel('Time')
ax1.set_ylabel('Sharpe ratio')
ax1.set_title('Difference in 12m rolling Sharpe Ratio and VIX\nRolling window optimization with 25% domestic, 2007-2024')
ax1.grid(True, linestyle='--', alpha=0.7)

# Create secondary axis for VIX
ax2 = ax1.twinx()

# Plot VIX if it exists
if 'VIX' in df_pivot.columns:
    vix_line = ax2.plot(df_pivot.index, df_pivot['VIX'],
                        color='red', label='VIX',
                        linestyle='--', linewidth=2)
    ax2.set_ylabel('VIX', color='red')
    ax2.tick_params(axis='y', labelcolor='red')
    # Set y-axis limits for VIX, 0, 35
    ax2.set_ylim(-10, 35)

# Combine legends
lines1, labels1 = ax1.get_legend_handles_labels()
if 'VIX' in df_pivot.columns:
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
else:
    ax1.legend(loc='upper left')

plt.tight_layout()
plt.show()





##### 12m rolling sharpe ratio Time series  full time period (2007-2023), 3 groups relative to no restriction group
### add MSCI_EM


import pandas as pd
import matplotlib.pyplot as plt

# Read the Excel file and create the DataFrame
sharpe_ratio = pd.read_excel('/Users/Stephen/Desktop/Learning_code/portfolio_diversification/rolling_window/Sharperatio_for_rolling.xlsx', '12m-Diffinchange_2nores-msciem')
df = pd.DataFrame(sharpe_ratio)

# Pivot the DataFrame to have time as index and columns as objects
df_pivot = df.pivot(index='Date', columns='Object', values='Value')

# Debug prints
print("Available columns:", df_pivot.columns.tolist())
print("\nFirst few rows of the data:")
print(df_pivot.head())

# Check VIX data specifically
if 'MSCI_EM' in df_pivot.columns:
    print("\nMSCI_EM data summary:")
    print(df_pivot['MSCI_EM'].describe())
    print("\nFirst few MSCI_EM values:")
    print(df_pivot['MSCI_EM'].head())
else:
    print("\nMSCI_EM column not found in the DataFrame")

# Create the plot
fig, ax1 = plt.subplots(figsize=(12, 6))

# Plot Sharpe ratios (excluding VIX)
for col in df_pivot.columns:
    if col != 'MSCI_EM':
        ax1.plot(df_pivot.index, df_pivot[col], label=col)

# Set up primary axis
ax1.set_xlabel('Time')
ax1.set_ylabel('Sharpe ratio')
ax1.set_title('Difference in 12m rolling Sharpe Ratio and MSCI EM\nRolling window optimization with 25% domestic, 2007-2024')
ax1.grid(True, linestyle='--', alpha=0.7)

# Create secondary axis for VIX
ax2 = ax1.twinx()

# Plot VIX if it exists
if 'MSCI_EM' in df_pivot.columns:
    MSCI_EM_line = ax2.plot(df_pivot.index, df_pivot['MSCI_EM'],
                        color='red', label='MSCI_EM',
                        linestyle='--', linewidth=2)
    ax2.set_ylabel('MSCI_EM', color='red')
    ax2.tick_params(axis='y', labelcolor='red')
    
    
    # Invert the y-axis
    ax2.invert_yaxis()
    
    
    # Set y-axis limits for MSCI_EM, 0, 35
    #ax2.set_ylim(-10, 35)

# Combine legends
lines1, labels1 = ax1.get_legend_handles_labels()
if 'MSCI_EM' in df_pivot.columns:
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
else:
    ax1.legend(loc='upper left')

plt.tight_layout()
plt.show()






##### 12m rolling sharpe ratio Time series  full time period (2007-2023), 3 groups relative to no restriction group
### add MSCI_world


import pandas as pd
import matplotlib.pyplot as plt

# Read the Excel file and create the DataFrame
sharpe_ratio = pd.read_excel('/Users/Stephen/Desktop/Learning_code/portfolio_diversification/rolling_window/Sharperatio_for_rolling.xlsx', '12m-Diffinchan_2nores-mscworld')
df = pd.DataFrame(sharpe_ratio)

# Pivot the DataFrame to have time as index and columns as objects
df_pivot = df.pivot(index='Date', columns='Object', values='Value')

# Debug prints
print("Available columns:", df_pivot.columns.tolist())
print("\nFirst few rows of the data:")
print(df_pivot.head())

# Check VIX data specifically
if 'MSCI_World' in df_pivot.columns:
    print("\nMSCI_World data summary:")
    print(df_pivot['MSCI_World'].describe())
    print("\nFirst few MSCI_World values:")
    print(df_pivot['MSCI_World'].head())
else:
    print("\nMSCI_World column not found in the DataFrame")

# Create the plot
fig, ax1 = plt.subplots(figsize=(12, 6))

# Plot Sharpe ratios (excluding MSCI World)
for col in df_pivot.columns:
    if col != 'MSCI_World':
        ax1.plot(df_pivot.index, df_pivot[col], label=col)

# Set up primary axis
ax1.set_xlabel('Time')
ax1.set_ylabel('Sharpe ratio')
ax1.set_title('Difference in 12m rolling Sharpe Ratio and MSCI World\nRolling window optimization with 25% domestic, 2007-2024')
ax1.grid(True, linestyle='--', alpha=0.7)

# Create secondary axis for VIX
ax2 = ax1.twinx()

# Plot VIX if it exists
if 'MSCI_World' in df_pivot.columns:
    MSCI_EM_line = ax2.plot(df_pivot.index, df_pivot['MSCI_World'],
                        color='red', label='MSCI_World',
                        linestyle='--', linewidth=2)
    ax2.set_ylabel('MSCI_World', color='red')
    ax2.tick_params(axis='y', labelcolor='red')
    
    
    # Invert the y-axis
    ax2.invert_yaxis()
    
    
    # Set y-axis limits for MSCI_World, 0, 35
    #ax2.set_ylim(-10, 35)

# Combine legends
lines1, labels1 = ax1.get_legend_handles_labels()
if 'MSCI_World' in df_pivot.columns:
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
else:
    ax1.legend(loc='upper left')

plt.tight_layout()
plt.show()










##### 24m rolling sharpe ratio Time series  full time period (2007-2023), all 4 groups relative to own idnex
sharpe_ratio = pd.read_excel('/Users/Stephen/Desktop/Learning_code/portfolio_diversification/rolling_window/Sharperatio_for_rolling.xlsx', '24mtime-changeinsharpe_rel2own')

df = pd.DataFrame(sharpe_ratio)


# Pivot the DataFrame to have time as index and objects as columns
df_pivot = df.pivot(index='Date', columns='Object', values='Value')

# Plot the data
fig, ax = plt.subplots()
df_pivot.plot(ax=ax)

# Add labels and title
ax.set_xlabel('Time')
ax.set_ylabel('Sharpe ratio')
ax.set_title('Difference in 24m rolling Sharpe ratio relative to domestic index \n Rolling window optimisation with 25% domestic, 2007-2024')

ax.legend(title='Scenario', fontsize='small', title_fontsize='small')

plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()





##### 24m rolling sharpe ratio Time series  full time period (2007-2023), 3 groups relative to no restriction group
sharpe_ratio = pd.read_excel('/Users/Stephen/Desktop/Learning_code/portfolio_diversification/rolling_window/Sharperatio_for_rolling.xlsx', '24mtime-Diffinchangeshar_2nores')

df = pd.DataFrame(sharpe_ratio)


# Pivot the DataFrame to have time as index and objects as columns
df_pivot = df.pivot(index='Date', columns='Object', values='Value')

# Plot the data
fig, ax = plt.subplots()
df_pivot.plot(ax=ax)

# Add labels and title
ax.set_xlabel('Time')
ax.set_ylabel('Sharpe ratio')
ax.set_title('Difference in 24m rolling Sharpe ratio relative to no geopolitical restriction\n Rolling window optimisation with 25% domestic, 2007-2024')


plt.ylim(-2.2, 1.05)

ax.legend(title='Scenario', fontsize='small', title_fontsize='small')

plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()





##### 24m rolling sharpe ratio Time series  full time period (2007-2023), 3 groups relative to no restriction group
### add VIX


import pandas as pd
import matplotlib.pyplot as plt

# Read the Excel file and create the DataFrame
sharpe_ratio = pd.read_excel('/Users/Stephen/Desktop/Learning_code/portfolio_diversification/rolling_window/Sharperatio_for_rolling.xlsx', '24mtime-Diffinchangeshar_2nores')
df = pd.DataFrame(sharpe_ratio)

# Pivot the DataFrame to have time as index and columns as objects
df_pivot = df.pivot(index='Date', columns='Object', values='Value')

# Debug prints
print("Available columns:", df_pivot.columns.tolist())
print("\nFirst few rows of the data:")
print(df_pivot.head())

# Check VIX data specifically
if 'VIX' in df_pivot.columns:
    print("\nVIX data summary:")
    print(df_pivot['VIX'].describe())
    print("\nFirst few VIX values:")
    print(df_pivot['VIX'].head())
else:
    print("\nVIX column not found in the DataFrame")

# Create the plot
fig, ax1 = plt.subplots(figsize=(12, 6))

# Plot Sharpe ratios (excluding VIX)
for col in df_pivot.columns:
    if col != 'VIX':
        ax1.plot(df_pivot.index, df_pivot[col], label=col)

# Set up primary axis
ax1.set_xlabel('Time')
ax1.set_ylabel('Sharpe ratio')
ax1.set_title('Difference in 24m Rolling Sharpe Ratio and VIX\nRolling window optimization with 25% domestic, 2007-2024')
ax1.grid(True, linestyle='--', alpha=0.7)

# Create secondary axis for VIX
ax2 = ax1.twinx()

# Plot VIX if it exists
if 'VIX' in df_pivot.columns:
    vix_line = ax2.plot(df_pivot.index, df_pivot['VIX'],
                        color='red', label='VIX',
                        linestyle='--', linewidth=2)
    ax2.set_ylabel('VIX', color='red')
    ax2.tick_params(axis='y', labelcolor='red')
    # Set y-axis limits for VIX
    ax2.set_ylim(0, 35)

# Combine legends
lines1, labels1 = ax1.get_legend_handles_labels()
if 'VIX' in df_pivot.columns:
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
else:
    ax1.legend(loc='upper left')

plt.tight_layout()
plt.show()




##### 25% in domestic, increase in sharpe ratio, full time period (2007-2023), break into 4 groups
sharpe_ratio = pd.read_excel('/Users/Stephen/Desktop/Learning_code/portfolio_diversification/rolling_window/Sharperatio_for_rolling.xlsx', '11index_bygroup')

df = pd.DataFrame(sharpe_ratio)

plt.figure(figsize=(8,6))

sns.boxplot(x='Scenario', y = 'Sharpe ratio', data =df, color = 'lightgray', fliersize=0)

sns.stripplot(x='Scenario', y="Sharpe ratio", data=df, color = 'black', alpha=0.7, jitter=True)

plt.ylim(-0.4, 1)

## show the plot
plt.title('Difference in Sharpe ratio relative to domestic index \n Rolling window optimisation with 25% domestic, 2007-2024')
plt.show()





##### 25% in domestic, increase in sharpe ratio, full time period (2007-2023), break into 4 groups
## here we want to see the difference in sharpe ratio for the 3 groups relative to geo-friends
sharpe_ratio = pd.read_excel('/Users/Stephen/Desktop/Learning_code/portfolio_diversification/rolling_window/Sharperatio_for_rolling.xlsx', '11index_rel2geofriend')

df = pd.DataFrame(sharpe_ratio)

plt.figure(figsize=(8,6))

sns.boxplot(x='Scenario', y = 'Sharpe ratio', data =df, color = 'lightgray', fliersize=0)

sns.stripplot(x='Scenario', y="Sharpe ratio", data=df, color = 'black', alpha=0.7, jitter=True)

plt.ylim(-0.6, 1)

## show the plot
plt.title('Difference in Sharpe ratio relative to diversifying to Geo-friends \n Rolling window optimisation with 25% domestic, 2007-2024')
plt.show()





##### 25% in domestic, increase in sharpe ratio, full time period (2007-2023), 
### check how much the sharpe increase relative to baseline
## cases: no restriction, exclude rivals, exclude distant, exclude close, exclude friend
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


sharpe_ratio = pd.read_excel('/Users/Stephen/Desktop/Learning_code/portfolio_diversification/rolling_window/Sharperatio_for_rolling.xlsx', '33index_bygroup')

df = pd.DataFrame(sharpe_ratio)

plt.figure(figsize=(8,6))

sns.boxplot(x='Scenario', y = 'Sharpe ratio', data =df, color = 'lightgray', fliersize=0)

sns.stripplot(x='Scenario', y="Sharpe ratio", data=df, color = 'black', alpha=0.7, jitter=True)

plt.ylim(-0.3, 0.8)

## show the plot
plt.title('Difference in Sharpe ratio relative to domestic index \n Rolling window optimisation with 25% domestic, 2007-2024')

# Rotate x-axis labels and split them into two lines
plt.xticks(rotation=45, ha='right')
labels = [
    'No restriction',
    'Exclude rivals',
    'Exclude\n less rivals',
    'Exclude\n less friends',
    'Exclude friend'
]
plt.gca().set_xticklabels(labels)

# Show the plot
plt.tight_layout()
plt.show()








##### 25% in domestic, decrease in sharpe ratio vs no restriction case, full time period (2007-2023), 
### check how much the sharpe decrease relative to no restriction case
## cases: exclude rivals, exclude distant, exclude close, exclude friend
sharpe_ratio = pd.read_excel('/Users/Stephen/Desktop/Learning_code/portfolio_diversification/rolling_window/Sharperatio_for_rolling.xlsx', '33index_rel2norest')

df = pd.DataFrame(sharpe_ratio)

plt.figure(figsize=(8,6))

sns.boxplot(x='Scenario', y = 'Sharpe ratio', data =df, color = 'lightgray', fliersize=0)

sns.stripplot(x='Scenario', y="Sharpe ratio", data=df, color = 'black', alpha=0.7, jitter=True)

plt.ylim(-0.35, 0.1)

## show the plot
plt.title('Difference in Sharpe ratio relative to no geopolitical restriction \n Rolling window optimisation with 25% domestic, 2007-2024')
plt.show()




##### 25% and 50% in domestic , difference in sharpe ratio relative to no restriction, full time period (2007-2023), exclude geo-rival vs no restriction 
sharpe_ratio = pd.read_excel('/Users/Stephen/Desktop/Learning_code/portfolio_diversification/rolling_window/Sharperatio_for_rolling.xlsx', '25_50%_excluderival_decre_2nore')

df = pd.DataFrame(sharpe_ratio)

plt.figure(figsize=(8,6))

sns.boxplot(x='Scenario', y = 'Sharpe ratio', data =df, color = 'lightgray', fliersize=0)

sns.stripplot(x='Scenario', y="Sharpe ratio", data=df, color = 'black', alpha=0.7, jitter=True)

plt.ylim(-0.35, 0)

## show the plot
plt.title('Difference in sharpe ratio relative to no geopolitical restriction \n Optimised portfolio with different % of domestic, 2007-2024')
plt.show()








