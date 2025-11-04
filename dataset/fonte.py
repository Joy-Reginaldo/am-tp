# =========================================================
# fonte_final_cmp263_user_friendly_full_v3.py - CMP263: Previsão de Vendas Amazon
# Dataset: amazon_products_sales_data_cleaned.csv
# Autores: Joice da Silva Reginaldo, Henrique Krausburg Correa
# =========================================================

import os
import warnings
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression

warnings.filterwarnings("ignore", category=FutureWarning)

# Tentar importar Plotly
try:
    import plotly.graph_objs as go
    import plotly.express as px
    PLOTLY_AVAILABLE = True
    print("[INFO] Plotly disponível: gráficos interativos habilitados.")
except ModuleNotFoundError:
    PLOTLY_AVAILABLE = False
    print("[INFO] Plotly não encontrado: gráficos interativos desabilitados.")

# =========================
# Configuração de caminhos
# =========================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(SCRIPT_DIR, "amazon_products_sales_data_cleaned.csv")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "../Result")
os.makedirs(OUTPUT_DIR, exist_ok=True)
HTML_REPORT_TECH = os.path.join(OUTPUT_DIR, "relatorio_tecnico.html")
HTML_REPORT_USER = os.path.join(OUTPUT_DIR, "relatorio_simplificado.html")

# =========================
# Funções auxiliares
# =========================
def safe_plot(plot_func, filename, *args, **kwargs):
    try:
        plot_func(*args, **kwargs)
        plt.savefig(os.path.join(OUTPUT_DIR, filename))
        plt.close()
        print(f"[INFO] Gráfico salvo: {filename}")
    except Exception as e:
        print(f"[ERRO] Não foi possível gerar {filename}: {e}")
        plt.close()

def plot_predictions_interactive(y_true, y_pred, output_file):
    if PLOTLY_AVAILABLE:
        fig = go.Figure()
        fig.add_trace(go.Scatter(y=y_true, mode='lines+markers', name='Real'))
        fig.add_trace(go.Scatter(y=y_pred, mode='lines+markers', name='Predito'))
        fig.update_layout(title="Previsões x Valores Reais",
                          xaxis_title="Índice", yaxis_title="Vendas")
        fig.write_html(output_file)
        print(f"[INFO] Gráfico interativo Plotly salvo em {output_file}")
    else:
        plt.figure(figsize=(10,6))
        plt.plot(y_true, marker='o', label='Real')
        plt.plot(y_pred, marker='x', label='Predito')
        plt.title("Previsões x Valores Reais")
        plt.xlabel("Índice")
        plt.ylabel("Vendas")
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_file.replace(".html", ".png"))
        plt.close()
        print(f"[INFO] Gráfico Matplotlib salvo em {output_file.replace('.html', '.png')}")

# =========================
# Carregar dataset
# =========================
if not os.path.exists(DATASET_PATH):
    raise FileNotFoundError(f"Arquivo CSV não encontrado: {DATASET_PATH}")

print("🔹 Carregando dataset...")
df = pd.read_csv(DATASET_PATH)
print(f" Dataset carregado: {df.shape[0]} linhas e {df.shape[1]} colunas.")

# =========================
# (i) Análise exploratória dos dados
# =========================
desc_stats = df.describe(include='all').to_html()
missing_data = df.isnull().sum().to_dict()

# Selecionar apenas colunas numéricas para histogramas, boxplots e correlação
num_cols = df.select_dtypes(include=[np.number]).columns.tolist()

# Histogramas das variáveis numéricas
for col in num_cols:
    safe_plot(lambda: sns.histplot(df[col], kde=True, color='skyblue'), f"hist_{col}.png")

# Boxplots para detecção de outliers
for col in num_cols:
    safe_plot(lambda: sns.boxplot(x=df[col], color='lightgreen'), f"box_{col}.png")

# Heatmap de correlação (apenas numéricas)
safe_plot(lambda: sns.heatmap(df[num_cols].corr(), annot=True, fmt=".2f", cmap="coolwarm"), "heatmap_correlation.png")

# =========================
# (ii) Pré-processamento dos dados
# =========================
df.fillna(0, inplace=True)
cat_cols = df.select_dtypes(include=['object']).columns.tolist()
for col in cat_cols:
    df[col] = df[col].astype(str)
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    print(f"   - Codificada coluna categórica: {col}")

target = "purchased_last_month"
X = df.drop(target, axis=1)
y = df[target]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# =========================
# (iii) Treinamento e validação dos modelos
# =========================
models = {
    "LinearRegression": LinearRegression(),
    "RandomForest": RandomForestRegressor(random_state=42, n_jobs=-1),
    "GradientBoosting": GradientBoostingRegressor(random_state=42)
}

try:
    from xgboost import XGBRegressor
    models["XGBoost"] = XGBRegressor(random_state=42, n_jobs=-1)
except ModuleNotFoundError:
    print("[INFO] XGBoost não disponível, ignorando.")

results = {}
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    results[name] = {
        "model": model,
        "y_pred": y_pred,
        "MAE": mean_absolute_error(y_test, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_test, y_pred)),
        "R2": r2_score(y_test, y_pred)
    }
    print(f" {name} avaliado: MAE={results[name]['MAE']:.2f}, RMSE={results[name]['RMSE']:.2f}, R2={results[name]['R2']:.4f}")

# Otimização RandomForest
param_grid = {'n_estimators':[100,200],'max_depth':[None,10,20],'min_samples_split':[2,5]}
grid = GridSearchCV(RandomForestRegressor(random_state=42, n_jobs=-1), param_grid, cv=3, scoring='r2', n_jobs=-1)
grid.fit(X_train, y_train)
best_rf = grid.best_estimator_
y_pred_best_rf = best_rf.predict(X_test)
print(f"🔹 RandomForest otimizado: R2={r2_score(y_test, y_pred_best_rf):.4f}")

# =========================
# (iv) Interpretação e análise crítica
# =========================
feature_importances = pd.DataFrame({
    'feature': X.columns,
    'importance': best_rf.feature_importances_
}).sort_values(by='importance', ascending=False)
safe_plot(lambda: sns.barplot(x='importance', y='feature', data=feature_importances, palette="viridis"), "feature_importance.png")

# =========================
# Comparação de métricas
# =========================
metrics_df = pd.DataFrame({name: {"MAE": r["MAE"], "RMSE": r["RMSE"], "R2": r["R2"]} for name, r in results.items()}).T
if PLOTLY_AVAILABLE:
    fig = px.bar(metrics_df.reset_index().melt(id_vars="index"), x="index", y="value", color="variable", barmode="group", title="Comparação de Métricas entre Modelos")
    fig.write_html(os.path.join(OUTPUT_DIR, "metrics_comparison.html"))
else:
    safe_plot(lambda: metrics_df.plot(kind='bar', figsize=(10,6)), "metrics_comparison.png")

plot_predictions_interactive(y_test, y_pred_best_rf, os.path.join(OUTPUT_DIR, "predictions_best_model.html"))

# =========================
# Relatório técnico com descrições detalhadas
# =========================
html_tech = f"""
<!DOCTYPE html>
<html lang='pt-BR'>
<head>
<meta charset='UTF-8'>
<title>Relatório Técnico Previsão de Vendas</title>
<style>body {{ font-family: Arial; margin: 20px; }} h1 {{color:#2F4F4F}} h2{{color:#2E8B57}} img{{margin-bottom:20px}}</style>
</head>
<body>
<h1>Previsão de Vendas Amazon (CMP263) - Técnico</h1>

<div class="box">
<h2>Objetivo do Projeto</h2>
<ul>
<li><strong>Objetivo:</strong> Prever quantos produtos serão comprados no mês</li>
<li><strong>Tipo de problema:</strong> Regressão</li>
<li><strong>Campo alvo:</strong> purchased_last_month</li>
<li><strong>Dificuldade:</strong> Difícil</li>
<li><strong>Questão de pesquisa:</strong> Quais os atributos mais relevantes que impactam na quantidade de vendas de um produto? Existe correlação?</li>
</ul>
</div>

<h2>(i) Análise exploratória dos dados</h2>
<p>Descrição estatística:</p>{desc_stats}
<p>Valores ausentes por coluna: {missing_data}</p>
<p>Heatmap de correlação (apenas variáveis numéricas): cores quentes indicam correlação positiva; frias indicam negativa.</p>
<img src='heatmap_correlation.png' width='700'><br>
"""

for col in num_cols:
    html_tech += f"<p>Histograma da variável <strong>{col}</strong>: distribuição dos valores, tendência e outliers.</p><img src='hist_{col}.png' width='700'><br>"
    html_tech += f"<p>Boxplot da variável <strong>{col}</strong>: mediana, quartis e possíveis valores extremos.</p><img src='box_{col}.png' width='700'><br>"

html_tech += f"""
<h2>(ii) Pré-processamento dos dados</h2>
<p>Preenchimento de valores ausentes, codificação de variáveis categóricas e normalização aplicados.</p>

<h2>(iii) Treinamento e validação dos modelos</h2>
<p>Modelos utilizados: {', '.join(results.keys())}</p>
<table border="1" cellpadding="5">
<tr><th>Modelo</th><th>MAE</th><th>RMSE</th><th>R2</th></tr>
"""
for name, r in results.items():
    html_tech += f"<tr><td>{name}</td><td>{r['MAE']:.2f}</td><td>{r['RMSE']:.2f}</td><td>{r['R2']:.4f}</td></tr>\n"
html_tech += "</table>"

html_tech += f"""
<h2>(iv) Interpretação e análise crítica</h2>
<p>O modelo RandomForest foi considerado o melhor baseado em R2. Os principais fatores que impactam as vendas são:</p>
<ul>
"""
for f, imp in feature_importances.values:
    html_tech += f"<li>{f}: importância {imp:.4f}</li>\n"
html_tech += "</ul>"

if PLOTLY_AVAILABLE:
    html_tech += "<p>Comparação de métricas interativa:</p><iframe src='metrics_comparison.html' width='900' height='600'></iframe>"
else:
    html_tech += "<p>Comparação de métricas:</p><img src='metrics_comparison.png' width='700'><br>"

html_tech += "<p>Previsões x valores reais (melhor modelo):</p><iframe src='predictions_best_model.html' width='900' height='600'></iframe>"
html_tech += "<p>Importância dos atributos:</p><img src='feature_importance.png' width='700'><br>"

with open(HTML_REPORT_TECH, "w", encoding="utf-8") as f:
    f.write(html_tech)

# =========================
# Relatório simplificado com descrições
# =========================
html_user = f"""
<!DOCTYPE html>
<html lang='pt-BR'>
<head><meta charset="UTF-8"><title>Relatório Simplificado Previsão de Vendas</title>
<style>body{{font-family:Arial;margin:20px;line-height:1.6;background:#f9f9f9}} h1{{color:#2F4F4F}} h2{{color:#2E8B57}} .box{{background:#fff;border:1px solid #ccc;padding:15px;margin-bottom:20px;border-radius:8px}} img,iframe{{margin-bottom:20px;border-radius:8px}}</style>
</head>
<body>
<h1>Previsão de Vendas Amazon</h1>

<div class="box">
<h2>Objetivo do Projeto</h2>
<p>Este projeto tem como objetivo prever quantos produtos serão comprados no mês, tratando-se de um problema de regressão com campo alvo <strong>purchased_last_month</strong>. A dificuldade é alta e a questão de pesquisa central é: quais atributos impactam mais na quantidade de vendas e como se correlacionam?</p>
</div>

<div class="box">
<h2>(i) Análise exploratória dos dados</h2>
<p>Resumo inicial dos dados e possíveis problemas, como valores ausentes ou outliers.</p>
<img src='heatmap_correlation.png' width='700'><br>
"""
for col in num_cols:
    html_user += f"<p>Histograma da variável <strong>{col}</strong>: distribuição e tendências.</p><img src='hist_{col}.png' width='700'><br>"
    html_user += f"<p>Boxplot da variável <strong>{col}</strong>: mediana, quartis e outliers.</p><img src='box_{col}.png' width='700'><br>"

html_user += f"""
<div class="box">
<h2>(ii) Pré-processamento dos dados</h2>
<p>Normalização, codificação de variáveis categóricas e preenchimento de valores ausentes aplicados.</p>
</div>

<div class="box">
<h2>(iii) Treinamento e validação dos modelos</h2>
<p>Modelos testados: {', '.join(results.keys())}</p>
<table border="1" cellpadding="5"><tr><th>Modelo</th><th>MAE</th><th>RMSE</th><th>R2</th></tr>
"""
for name, r in results.items():
    html_user += f"<tr><td>{name}</td><td>{r['MAE']:.2f}</td><td>{r['RMSE']:.2f}</td><td>{r['R2']:.4f}</td></tr>\n"
html_user += "</table></div>"

html_user += f"""
<div class="box">
<h2>(iv) Interpretação e análise crítica</h2>
<p>O modelo RandomForest foi considerado o melhor com base em R2. Os fatores que mais influenciam as vendas são:</p>
<ul>
"""
for f, imp in feature_importances.values:
    html_user += f"<li>{f}: importância {imp:.4f}</li>\n"
html_user += "</ul>"
html_user += "<p>Gráfico das previsões do próximo mês:</p><iframe src='predictions_best_model.html' width='900' height='600'></iframe>"
html_user += "<p>Importância dos atributos:</p><img src='feature_importance.png' width='700'><br>"
html_user += "</body></html>"

with open(HTML_REPORT_USER, "w", encoding="utf-8") as f:
    f.write(html_user)

print(f"\n Relatório técnico salvo em: {HTML_REPORT_TECH}")
print(f" Relatório simplificado salvo em: {HTML_REPORT_USER}")
print("\n Execução finalizada com sucesso!")
