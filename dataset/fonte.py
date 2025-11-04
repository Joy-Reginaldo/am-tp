# =========================================================
# fonte_final_cmp263_final.py - CMP263: Previsão de Vendas Amazon
# Projeto Final: Aplicação de Boas Práticas em Machine Learning
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
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

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
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "Result")
os.makedirs(OUTPUT_DIR, exist_ok=True)
HTML_REPORT_TECH = os.path.join(OUTPUT_DIR, "relatorio_tecnico_final.html")
HTML_REPORT_USER = os.path.join(OUTPUT_DIR, "relatorio_simplificado_final.html")

# =========================
# Funções auxiliares
# =========================
def safe_plot(plot_func, filename, *args, **kwargs):
    """Salva um gráfico Matplotlib com tratamento de erro."""
    try:
        plot_func(*args, **kwargs)
        plt.savefig(os.path.join(OUTPUT_DIR, filename))
        plt.close()
        print(f"[INFO] Gráfico salvo: {filename}")
    except Exception as e:
        print(f"[ERRO] Não foi possível gerar {filename}: {e}")
        plt.close()

def plot_predictions_interactive(y_true, y_pred, output_file):
    """Gera gráfico interativo Plotly ou estático Matplotlib das previsões."""
    if PLOTLY_AVAILABLE:
        fig = go.Figure()
        # Usar um subconjunto para Plotly se o dataset for muito grande
        sample_size = min(len(y_true), 1000) 
        indices = np.random.choice(len(y_true), sample_size, replace=False)
        
        fig.add_trace(go.Scatter(y=y_true.iloc[indices], mode='lines+markers', name='Real'))
        fig.add_trace(go.Scatter(y=y_pred[indices], mode='lines+markers', name='Predito'))
        fig.update_layout(title="Previsões x Valores Reais (Amostra)",
                          xaxis_title="Índice", yaxis_title="Vendas")
        fig.write_html(output_file)
        print(f"[INFO] Gráfico interativo Plotly salvo em {output_file}")
    else:
        plt.figure(figsize=(10,6))
        plt.plot(y_true.values, marker='o', label='Real')
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

# =========================
# (i) Análise exploratória dos dados (Antes do Pré-processamento)
# =========================

# 1. Remoção de linhas com valor alvo ausente (Crítica: Nunca mexer no valor do campo alvo)
# Para um problema de regressão, não podemos imputar o valor alvo.
target = "purchased_last_month"
df.dropna(subset=[target], inplace=True)
print(f" Dataset carregado após remover {target} ausentes: {df.shape[0]} linhas e {df.shape[1]} colunas.")

desc_stats = df.describe(include='all').to_html()
missing_data = df.isnull().sum().to_dict()
missing_data_html = "<ul>" + "".join([f"<li>{k}: {v}</li>" for k, v in missing_data.items() if v > 0]) + "</ul>"

# Identificar colunas para diferentes tipos de pré-processamento
num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
# Remover o target da lista de colunas numéricas para análise
if target in num_cols:
    num_cols.remove(target) 

# Identificar colunas categóricas (object) que não são identificadores/textos longos
# Assumindo que 'category' é a única categórica nominal útil.
# 'product_id', 'title', 'date_added' são considerados para descarte ou tratamento especial.
cat_cols = ['category'] 
# Colunas a serem descartadas ou tratadas com métodos mais avançados (fora do escopo desta correção simples)
cols_to_drop = ['product_id', 'title', 'date_added'] 

# Histogramas das variáveis numéricas (Apenas para as colunas numéricas que não são o target)
for col in num_cols:
    safe_plot(lambda: sns.histplot(df[col], kde=True, color='skyblue'), f"hist_{col}.png")

# Boxplots para detecção de outliers
for col in num_cols:
    safe_plot(lambda: sns.boxplot(x=df[col], color='lightgreen'), f"box_{col}.png")

# Heatmap de correlação (apenas numéricas)
all_num_cols = num_cols + [target]
safe_plot(lambda: sns.heatmap(df[all_num_cols].corr(), annot=True, fmt=".2f", cmap="coolwarm"), "heatmap_correlation.png")

# =========================
# (ii) Pré-processamento dos dados (Uso de Pipeline e ColumnTransformer)
# =========================

# Separação de X e y
# Garante que apenas colunas que existem no DataFrame sejam incluídas na lista de descarte
existing_cols_to_drop = [col for col in cols_to_drop if col in df.columns]
X = df.drop(columns=[target] + existing_cols_to_drop)
y = df[target]

# Separação de treino e teste (Pré-processamento só pode ser feito após o split)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Definição dos transformadores para o ColumnTransformer
# 1. Numéricas: Imputação por Mediana (menos sensível a outliers que a média) e Scaling
numeric_features = X_train.select_dtypes(include=np.number).columns.tolist()
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# 2. Categóricas: Imputação por valor mais frequente e One-Hot Encoding
categorical_features = X_train.select_dtypes(include='object').columns.tolist()
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Criação do pré-processador
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ],
    remainder='passthrough' # Manter colunas não processadas (ex: date_added se não for descartada)
)

# =========================
# (iii) Treinamento e validação dos modelos (Uso de Pipeline)
# =========================
models_raw = {
    "LinearRegression": LinearRegression(),
    "RandomForest": RandomForestRegressor(random_state=42, n_jobs=-1),
    "GradientBoosting": GradientBoostingRegressor(random_state=42)
}

# Removido XGBoost para manter o foco nos modelos principais do curso

results = {}
best_model_name = ""
best_r2 = -np.inf

for name, model in models_raw.items():
    # Criação do Pipeline: Pré-processamento + Modelo
    full_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                    ('regressor', model)])
    
    full_pipeline.fit(X_train, y_train)
    y_pred = full_pipeline.predict(X_test)
    
    r2 = r2_score(y_test, y_pred)
    
    results[name] = {
        "model": full_pipeline,
        "y_pred": y_pred,
        "MAE": mean_absolute_error(y_test, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_test, y_pred)),
        "R2": r2
    }
    print(f" {name} avaliado: MAE={results[name]['MAE']:.2f}, RMSE={results[name]['RMSE']:.2f}, R2={results[name]['R2']:.4f}")
    
    if r2 > best_r2:
        best_r2 = r2
        best_model_name = name

# Otimização RandomForest (Exemplo de GridSearch com Pipeline)
# Nota: O GridSearch deve ser aplicado ao Pipeline completo para evitar vazamento.
print("\n🔹 Otimizando o melhor modelo (RandomForest, se disponível)...")
if "RandomForest" in models_raw:
    rf_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                  ('regressor', RandomForestRegressor(random_state=42, n_jobs=-1))])
    
    # Otimização de hiperparâmetros (reduzida para agilizar)
    param_grid = {
        'regressor__n_estimators': [50, 100],
        'regressor__max_depth': [5, 10]
    }
    
    grid = GridSearchCV(rf_pipeline, param_grid, cv=3, scoring='r2', n_jobs=-1)
    grid.fit(X_train, y_train)
    
    best_rf_pipeline = grid.best_estimator_
    y_pred_best_rf = best_rf_pipeline.predict(X_test)
    
    # Atualiza resultados com o modelo otimizado
    results["RandomForest_Otimizado"] = {
        "model": best_rf_pipeline,
        "y_pred": y_pred_best_rf,
        "MAE": mean_absolute_error(y_test, y_pred_best_rf),
        "RMSE": np.sqrt(mean_squared_error(y_test, y_pred_best_rf)),
        "R2": r2_score(y_test, y_pred_best_rf)
    }
    print(f"🔹 RandomForest otimizado: R2={results['RandomForest_Otimizado']['R2']:.4f}")
    
    # Se o otimizado for melhor, ele se torna o "melhor modelo" para relatórios
    if results['RandomForest_Otimizado']['R2'] > best_r2:
        best_r2 = results['RandomForest_Otimizado']['R2']
        best_model_name = "RandomForest_Otimizado"

best_model_result = results[best_model_name]
best_model = best_model_result['model']
y_pred_best = best_model_result['y_pred']

# =========================
# (iv) Interpretação e análise crítica
# =========================

feature_importances = pd.DataFrame()
# Feature importance só é aplicável a modelos baseados em árvore como RandomForest
if "RandomForest" in best_model_name:
    # Extrair feature importances do modelo dentro do pipeline
    regressor = best_model.named_steps['regressor']
    
    # Obter nomes das features após o OneHotEncoding
    # Nota: get_feature_names_out é o método correto para ColumnTransformer
    processed_feature_names = preprocessor.get_feature_names_out()
    
    # Filtrar apenas as features que foram usadas no regressor (após o 'preprocessor__')
    # O regressor recebe todas as features processadas.
    # O número de importâncias deve ser igual ao número de features processadas.
    
    feature_importances = pd.DataFrame({
        'feature': processed_feature_names,
        'importance': regressor.feature_importances_
    }).sort_values(by='importance', ascending=False).head(10) # Top 10

    safe_plot(lambda: sns.barplot(x='importance', y='feature', data=feature_importances, palette="viridis"), "feature_importance_final.png")
    
    # Salvar a lista de importâncias para o relatório
    feature_importance_list = feature_importances.values.tolist()
else:
    # Para LinearRegression, não há feature_importance simples.
    feature_importance_list = [["Não aplicável", 0.0]]


# =========================
# Comparação de métricas
# =========================
metrics_df = pd.DataFrame({name: {"MAE": r["MAE"], "RMSE": r["RMSE"], "R2": r["R2"]} for name, r in results.items()}).T
metrics_df = metrics_df.sort_values(by='R2', ascending=False)

if PLOTLY_AVAILABLE:
    fig = px.bar(metrics_df.reset_index().melt(id_vars="index"), x="index", y="value", color="variable", barmode="group", title="Comparação de Métricas entre Modelos")
    fig.write_html(os.path.join(OUTPUT_DIR, "metrics_comparison_final.html"))
else:
    safe_plot(lambda: metrics_df.plot(kind='bar', figsize=(10,6)), "metrics_comparison_final.png")

plot_predictions_interactive(y_test, y_pred_best, os.path.join(OUTPUT_DIR, "predictions_best_model_final.html"))

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
<h1>Previsão de Vendas Amazon (CMP263) - Relatório Técnico</h1>

<div class="box">
<h2>Objetivo do Projeto</h2>
<ul>
<li><strong>Objetivo:</strong> Prever quantos produtos serão comprados no mês</li>
<li><strong>Tipo de problema:</strong> Regressão</li>
<li><strong>Campo alvo:</strong> {target}</li>
<li><strong>Melhor Modelo:</strong> {best_model_name}</li>
</ul>
</div>

<h2>(i) Análise exploratória dos dados</h2>
<p><strong>Observação:</strong> Linhas com valor alvo ({target}) ausente foram removidas para garantir a integridade do treino.</p>
<p>Descrição estatística (após remoção de target ausente):</p>{desc_stats}
<p>Valores ausentes restantes por coluna (tratados no Pipeline):</p>{missing_data_html}
<p>Heatmap de correlação (apenas variáveis numéricas, incluindo o target):</p>
<img src='heatmap_correlation.png' width='700'><br>
"""

# Inclusão dos gráficos de análise exploratória
for col in num_cols:
    html_tech += f"<p>Histograma da variável <strong>{col}</strong>.</p><img src='hist_{col}.png' width='700'><br>"
    html_tech += f"<p>Boxplot da variável <strong>{col}</strong>.</p><img src='box_{col}.png' width='700'><br>"

html_tech += f"""
<h2>(ii) Pré-processamento dos dados</h2>
<p>O pré-processamento é realizado DENTRO do pipeline de Machine Learning, APÓS a separação entre treino e teste, o que garante que as informações do conjunto de teste não "vazem" para o treino.</p>
<ul>
    <li>**Tratamento de Ausentes:** Imputação por Mediana para numéricas e por valor mais frequente para categóricas.</li>
    <li>**Codificação Categórica:** Uso de One-Hot Encoding (OHE) para a coluna 'category'. Colunas de texto/ID foram descartadas.</li>
    <li>**Normalização:** StandardScaler aplicado apenas aos dados numéricos imputados.</li>
</ul>

<h2>(iii) Treinamento e validação dos modelos</h2>
<p>Modelos testados (todos dentro de um Pipeline): {', '.join(results.keys())}</p>
<table border="1" cellpadding="5">
<tr><th>Modelo</th><th>MAE</th><th>RMSE</th><th>R2</th></tr>
"""
for name, r in metrics_df.iterrows():
    html_tech += f"<tr><td>{name}</td><td>{r['MAE']:.2f}</td><td>{r['RMSE']:.2f}</td><td>{r['R2']:.4f}</td></tr>\n"
html_tech += "</table>"

html_tech += f"""
<h2>(iv) Interpretação e análise crítica</h2>
<p>O modelo **{best_model_name}** foi o que apresentou o melhor desempenho (R2={best_r2:.4f}).</p>

<h3>Importância dos Atributos (Top {len(feature_importances)} - {best_model_name})</h3>
<p>Os principais fatores que impactam as vendas, de acordo com a importância do atributo no modelo de árvore:</p>
<ul>
"""
for f, imp in feature_importance_list:
    html_tech += f"<li>{f}: importância {imp:.4f}</li>\n"
html_tech += "</ul>"

if "RandomForest" in best_model_name:
    html_tech += "<p>Gráfico de Importância dos Atributos:</p><img src='feature_importance_final.png' width='700'><br>"


if PLOTLY_AVAILABLE:
    html_tech += "<p>Comparação de métricas interativa:</p><iframe src='metrics_comparison_final.html' width='900' height='600'></iframe>"
else:
    html_tech += "<p>Comparação de métricas:</p><img src='metrics_comparison_final.png' width='700'><br>"

html_tech += "<p>Previsões x valores reais (melhor modelo):</p><iframe src='predictions_best_model_final.html' width='900' height='600'></iframe>"

html_tech += "</body></html>"

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
<p>Este projeto tem como objetivo prever quantos produtos serão comprados no mês, tratando-se de um problema de regressão com campo alvo <strong>{target}</strong>. A questão de pesquisa central é: quais atributos impactam mais na quantidade de vendas e como se correlacionam?</p>
</div>

<div class="box">
<h2>(i) Análise exploratória dos dados</h2>
<p>Resumo inicial dos dados e possíveis problemas, como valores ausentes ou outliers. O gráfico abaixo mostra a correlação entre as variáveis numéricas:</p>
<img src='heatmap_correlation.png' width='700'><br>
"""
for col in num_cols:
    html_user += f"<p>Histograma da variável <strong>{col}</strong>: distribuição e tendências.</p><img src='hist_{col}.png' width='700'><br>"
    html_user += f"<p>Boxplot da variável <strong>{col}</strong>: mediana, quartis e outliers.</p><img src='box_{col}.png' width='700'><br>"

html_user += f"""
<div class="box">
<h2>(ii) Pré-processamento dos dados</h2>
<p>Para garantir a validade do modelo, o pré-processamento (normalização, codificação de variáveis categóricas e tratamento de ausentes) foi aplicado **após** a separação do conjunto de treino e teste (Pipeline).</p>
</div>

<div class="box">
<h2>(iii) Treinamento e validação dos modelos</h2>
<p>Modelos testados: {', '.join(results.keys())}</p>
<table border="1" cellpadding="5"><tr><th>Modelo</th><th>MAE</th><th>RMSE</th><th>R2</th></tr>
"""
for name, r in metrics_df.iterrows():
    html_user += f"<tr><td>{name}</td><td>{r['MAE']:.2f}</td><td>{r['RMSE']:.2f}</td><td>{r['R2']:.4f}</td></tr>\n"
html_user += "</table></div>"

html_user += f"""
<div class="box">
<h2>(iv) Interpretação e análise crítica</h2>
<p>O modelo **{best_model_name}** foi considerado o melhor com base em R2. Os fatores que mais influenciam as vendas são:</p>
<ul>
"""
for f, imp in feature_importance_list:
    html_user += f"<li>{f}: importância {imp:.4f}</li>\n"
html_user += "</ul>"
html_user += "<p>Gráfico das previsões do melhor modelo:</p><iframe src='predictions_best_model_final.html' width='900' height='600'></iframe>"
html_user += "<p>Importância dos atributos:</p><img src='feature_importance_final.png' width='700'><br>"
html_user += "</body></html>"

with open(HTML_REPORT_USER, "w", encoding="utf-8") as f:
    f.write(html_user)

print(f"\n Relatório técnico final salvo em: {HTML_REPORT_TECH}")
print(f" Relatório simplificado final salvo em: {HTML_REPORT_USER}")
print("\n Execução finalizada com sucesso!")