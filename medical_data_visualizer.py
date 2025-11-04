import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# 1: Import data
df = pd.read_csv('medical_examination.csv')


# 2: Add 'overweight' column
bmi = df['weight']/np.square(df['height']/100)
df['overweight'] = np.where(bmi>25,1,0)

# 3: Normalize data by making 0 always good and 1 always bad
# If the value of 'cholesterol' or 'gluc' is 1, make the value 0. If the value is more than 1, make the value 1.
df['cholesterol'] = np.where(df['cholesterol']>1,1,0)
df['gluc']= np.where(df['gluc']>1,1,0)

# 4:Draw Categorical Plot
def draw_cat_plot():

    # 5: Create DataFrame for cat plot using `pd.melt` using just the values from 'cholesterol', 'gluc', 'smoke', 'alco', 'active', and 'overweight'.
    df_cat = pd.melt(df, id_vars=['cardio'], value_vars=['cholesterol', 'gluc', 'smoke', 'alco', 'active','overweight'], var_name= 'variable', value_name = 'value')


    # 6: Group and reformat the data to split it by 'cardio'. Show the counts of each feature. You will have to rename one of the columns for the catplot to work correctly.
    df_cat = df_cat.groupby(['cardio','variable','value']).size()
    df_cat = df_cat.reset_index(name= 'total')
    

    # 7: Draw the catplot with 'sns.catplot()'
    graph = sns.catplot(data = df_cat, x='variable', y='total', hue = 'value', col = 'cardio', kind = 'bar')


    # 8: Get the figure for the output and store it in the fig variable.
    fig = graph.fig


    # 9: Do not modify the next two lines
    fig.savefig('catplot.png')
    return fig


# 10: Draw Heat Map
def draw_heat_map():

    # 11: Clean the data
    df_heat = df[(df['ap_lo'] <= df['ap_hi']) & (df['height'] >= df['height'].quantile(0.025)) & (df['height'] <= df['height'].quantile(0.975)) & (df['weight'] >= df['weight'].quantile(0.025)) & (df['weight'] <= df['weight'].quantile(0.975))]


    # 12: Calculate the correlation matrix
    corr = df_heat.corr(method = "pearson")

    # 13: Generate a mask for the upper triangle
    mask = np.triu(np.ones_like(corr))



    # 14: Set up the matplotlib figure
    fig, ax = plt.subplots(figsize=(10,10))

    # 15: Draw the heatmap with 'sns.heatmap()'
    ax = sns.heatmap(corr, mask=mask, fmt=".1f", center=0.0, vmin=-0.16, vmax= 0.3, annot = True,  square=True, linewidths=0.5, cbar_kws={"shrink": 0.5,"ticks": np.linspace(-0.08,0.24,5)}, ax=ax)



    # 16: Do not modify the next two lines
    fig.savefig('heatmap.png')
    return fig


