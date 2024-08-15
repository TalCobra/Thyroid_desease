# -*- coding: utf-8 -*-
"""
Created on Thu Jul 25 16:03:36 2024

@author: Matheus Henrique Dizaro Miyamoto
"""
#%% Instalando os pacotes

!pip install pandas
!pip install numpy
!pip install matplotlib
!pip install -U seaborn
!pip install plotly
!pip install scipy
!pip install scikit-learn
!pip install pingouin
!pip install statsmodels
!pip install factor_analyzer
!pip install prince

#%% Importando os pacotes

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import pingouin as pg
import plotly.express as px 
import plotly.io as pio
pio.renderers.default='browser'
from factor_analyzer import FactorAnalyzer
from factor_analyzer.factor_analyzer import calculate_bartlett_sphericity
from scipy.stats import chi2_contingency
import statsmodels.api as sm
import plotly.graph_objects as go
import prince

#%% Importando o banco de dados

disease_data = pd.read_csv('Thyroid_Diff.csv', delimiter=',')
# Fonte: https://www.kaggle.com/datasets/jainaru/thyroid-disease-data e https://link.springer.com/article/10.1007/s00405-023-08299-w
disease_data.info()

disease_data_quali = disease_data[['Gender',
                                   'Smoking',
                                   'Hx Smoking',
                                   'Hx Radiothreapy',
                                   'Thyroid Function',
                                   'Physical Examination',
                                   'Adenopathy',
                                   'Pathology',
                                   'Focality',
                                   'Risk',
                                   'T',
                                   'N',
                                   'M',
                                   'Stage',
                                   'Response',
                                   'Recurred']]
print(disease_data_quali['Gender'].value_counts())
print(disease_data_quali['Smoking'].value_counts())
print(disease_data_quali['Hx Smoking'].value_counts())
print(disease_data_quali['Hx Radiothreapy'].value_counts())
print(disease_data_quali['Thyroid Function'].value_counts())
print(disease_data_quali['Physical Examination'].value_counts())
print(disease_data_quali['Adenopathy'].value_counts())
print(disease_data_quali['Pathology'].value_counts())
print(disease_data_quali['Focality'].value_counts())
print(disease_data_quali['Risk'].value_counts())
print(disease_data_quali['T'].value_counts())
print(disease_data_quali['N'].value_counts())
print(disease_data_quali['M'].value_counts())
print(disease_data_quali['Stage'].value_counts())
print(disease_data_quali['Response'].value_counts())
print(disease_data_quali['Recurred'].value_counts())
#%% Teste qui² de associação

tabela_mca_1 = chi2_contingency(pd.crosstab(disease_data_quali['Smoking'], disease_data_quali['Gender']))
tabela_mca_2 = chi2_contingency(pd.crosstab(disease_data_quali['Smoking'], disease_data_quali['Hx Smoking']))
tabela_mca_3 = chi2_contingency(pd.crosstab(disease_data_quali['Smoking'], disease_data_quali['Hx Radiothreapy']))
tabela_mca_4 = chi2_contingency(pd.crosstab(disease_data_quali['Smoking'], disease_data_quali['Thyroid Function']))
tabela_mca_5 = chi2_contingency(pd.crosstab(disease_data_quali['Smoking'], disease_data_quali['Adenopathy']))
tabela_mca_6 = chi2_contingency(pd.crosstab(disease_data_quali['Smoking'], disease_data_quali['Pathology']))
tabela_mca_7 = chi2_contingency(pd.crosstab(disease_data_quali['Smoking'], disease_data_quali['Focality']))
tabela_mca_8 = chi2_contingency(pd.crosstab(disease_data_quali['Smoking'], disease_data_quali['Risk']))
tabela_mca_9 = chi2_contingency(pd.crosstab(disease_data_quali['Smoking'], disease_data_quali['T']))
tabela_mca_10 = chi2_contingency(pd.crosstab(disease_data_quali['Smoking'], disease_data_quali['N']))
tabela_mca_11 = chi2_contingency(pd.crosstab(disease_data_quali['Smoking'], disease_data_quali['M']))
tabela_mca_12 = chi2_contingency(pd.crosstab(disease_data_quali['Smoking'], disease_data_quali['Stage']))
tabela_mca_13 = chi2_contingency(pd.crosstab(disease_data_quali['Smoking'], disease_data_quali['Response']))
tabela_mca_14 = chi2_contingency(pd.crosstab(disease_data_quali['Smoking'], disease_data_quali['Recurred']))

print(f"p-valor da estatística: {round(tabela_mca_1[1], 4)}")
print(f"p-valor da estatística: {round(tabela_mca_2[1], 4)}")
print(f"p-valor da estatística: {round(tabela_mca_3[1], 4)}")
print(f"p-valor da estatística: {round(tabela_mca_4[1], 4)}")
print(f"p-valor da estatística: {round(tabela_mca_5[1], 4)}")
print(f"p-valor da estatística: {round(tabela_mca_6[1], 4)}")
print(f"p-valor da estatística: {round(tabela_mca_7[1], 4)}")
print(f"p-valor da estatística: {round(tabela_mca_8[1], 4)}")
print(f"p-valor da estatística: {round(tabela_mca_9[1], 4)}")
print(f"p-valor da estatística: {round(tabela_mca_10[1], 4)}")
print(f"p-valor da estatística: {round(tabela_mca_11[1], 4)}")
print(f"p-valor da estatística: {round(tabela_mca_12[1], 4)}")
print(f"p-valor da estatística: {round(tabela_mca_13[1], 4)}")
print(f"p-valor da estatística: {round(tabela_mca_14[1], 4)}")

disease_data_quali.drop(columns=['Thyroid Function'],inplace=True)
#%%ACM

mca = prince.MCA(n_components=3).fit(disease_data_quali)
quant_dim = mca.J_ - mca.K_
print(mca.total_inertia_/quant_dim)
tabela_autovalores = mca.eigenvalues_summary
coord_pad = mca.column_coordinates(disease_data_quali)/np.sqrt(mca.eigenvalues_)

coord_obs = mca.row_coordinates(disease_data_quali)
coord_obs.rename(columns={0: 'dim1_acm', 1: 'dim2_acm',2: 'dim3_acm'}, inplace=True)
print(coord_obs)
#%% Plotando mapa perceptual

# Primeiro passo: gerar um DataFrame detalhado

chart = coord_pad.reset_index()

var_chart = pd.Series(chart['index'].str.split('_', expand=True).iloc[:,0])

nome_categ=[]
for col in disease_data_quali:
    nome_categ.append(disease_data_quali[col].sort_values(ascending=True).unique())
    categorias = pd.DataFrame(nome_categ).stack().reset_index()

chart_df_mca = pd.DataFrame({'categoria': chart['index'],
                             'obs_x': chart[0],
                             'obs_y': chart[1],
                             'obs_z': chart[2],
                             'variavel': var_chart,
                             'categoria_id': categorias[0]})

# Segundo passo: gerar o gráfico de pontos

fig = px.scatter_3d(chart_df_mca, 
                    x='obs_x', 
                    y='obs_y', 
                    z='obs_z',
                    color='variavel',
                    text=chart_df_mca.categoria_id)
fig.write_html('Scaterplott_3D.html')

# Abrir o arquivo HTML no navegador
import webbrowser
webbrowser.open('Scaterplott_3D.html')

#%%
