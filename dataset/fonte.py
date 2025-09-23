import os
import warnings
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
matplotlib.use('Agg')  # Evita problemas com Tkinter
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ============================
# Supressão de warnings do Seaborn
# ============================
warnings.filterwarnings("ignore", category=FutureWarning)

# ============================
# Função auxiliar para gráficos
# ============================
def safe_plot(plot_func, filename, *args, **kwargs):
    """Executa função de plotagem e salva, sem quebrar o script."""
    try:
        plot_func(*args, **kwargs)
        plt.savefig(os.path.join(output_dir, filename))
        plt.close()
    except Exception as e:
        print(f"Não foi possível gerar {filename}: {e}")
        plt.close()

# ============================
# 1. Configuração de caminhos
# ============================
script_dir = os.path.dirname(os.path.abspath(__file__))  # pasta do script
csv_path = os.path.join(script_dir, "amazon_products_sales_data_cleaned.csv")  # CSV na mesma pasta
output_dir = os.path.join(script_dir, "Result")
os.makedirs(output_dir, exist_ok=True)

# ============================
# 2. Carregar dataset
# ============================
try:
    df = pd.read_csv(csv_path)
except FileNotFoundError:
    raise FileNotFoundError(f"Arquivo CSV não encontrado: {csv_path}")

df.fillna(0, inplace=True)  # Garantir que não existam valores nulos
print(f"Dataset carregado: {df.shape[0]} linhas x {df.shape[1]} colunas")

# Salvar dataset completo em CSV
df.to_csv(os.path.join(output_dir, "dataset_full.csv"), index=False)

# Salvar dataset em Excel, se possível
try:
    df.to_excel(os.path.join(output_dir, "dataset_full.xlsx"), index=False)
except ModuleNotFoundError:
    print("Módulo 'openpyxl' não encontrado. Arquivo Excel não será gerado.")

# ============================
# 3. Estatísticas
# ============================
summary = df.describe(include="all")
summary.to_csv(os.path.join(output_dir, "dataset_summary.csv"))
print("\n=== Estatísticas do Dataset ===")
print(summary)

# ============================
# 4. Visualizações exploratórias
# ============================

# Distribuição de vendas no último mês
if "purchased_last_month" in df.columns:
    plt.figure(figsize=(8,5))
    safe_plot(sns.histplot, "dist_purchased_last_month.png",
              data=df["purchased_last_month"], bins=30, color="skyblue", kde=False)
    print("\n=== Distribuição de compras no último mês ===")
    print(df["purchased_last_month"].describe())

# Distribuição de ratings
if "product_rating" in df.columns:
    plt.figure(figsize=(8,5))
    safe_plot(sns.histplot, "dist_ratings.png",
              data=df["product_rating"], bins=20, color="green", kde=True)
    print("\n=== Distribuição de ratings ===")
    print(df["product_rating"].describe())

# Top 10 categorias
if "product_category" in df.columns:
    plt.figure(figsize=(10,6))
    top_categories = df["product_category"].value_counts().head(10)
    safe_plot(sns.barplot, "top10_categorias.png",
              x=top_categories.values, y=top_categories.index, dodge=False)
    print("\n=== Top 10 Categorias de Produtos ===")
    print(top_categories)

# Preço x Avaliação
if "discounted_price" in df.columns and "product_rating" in df.columns:
    plt.figure(figsize=(8,5))
    safe_plot(sns.scatterplot, "scatter_preco_rating.png",
              data=df, x="discounted_price", y="product_rating", alpha=0.5)

# Correlação entre variáveis numéricas
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
if len(numeric_cols) > 1:
    plt.figure(figsize=(12,8))
    corr = df[numeric_cols].corr()
    safe_plot(sns.heatmap, "correlation_heatmap.png", corr, annot=True, cmap="coolwarm", fmt=".2f")

# ============================
# 5. Preparação para modelo
# ============================
target = "purchased_last_month"
if target not in df.columns:
    raise ValueError(f"A coluna alvo '{target}' não existe no dataset!")

num_features = ["product_rating", "total_reviews", "discounted_price", "original_price", "discount_percentage"]
cat_features = ["product_category", "is_best_seller", "is_sponsored", "has_coupon", "buy_box_availability"]

num_features = [f for f in num_features if f in df.columns]
cat_features = [f for f in cat_features if f in df.columns]

X = df[num_features + cat_features]
y = df[target]

# One-Hot Encoding
X = pd.get_dummies(X, columns=cat_features, drop_first=True)

# ============================
# 6. Modelo preditivo
# ============================
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

# ============================
# 7. Avaliação
# ============================
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("\n=== Avaliação do Modelo ===")
print(f"MAE  (Erro Absoluto Médio): {mae:.2f}")
print(f"RMSE (Raiz do Erro Quadrático Médio): {rmse:.2f}")
print(f"R²   (Coeficiente de Determinação): {r2:.2f}")

with open(os.path.join(output_dir, "model_evaluation.txt"), "w", encoding="utf-8") as f:
    f.write("Avaliação do modelo:\n")
    f.write(f"MAE  (Erro Absoluto Médio): {mae:.2f}\n")
    f.write(f"RMSE (Raiz do Erro Quadrático Médio): {rmse:.2f}\n")
    f.write(f"R²   (Coeficiente de Determinação): {r2:.2f}\n")

# ============================
# 8. Importância dos atributos
# ============================
importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=True)

plt.figure(figsize=(10,8))
safe_plot(importances.tail(20).plot, "feature_importance.png", kind="barh", color="teal")

# ============================
# 9. Top 10 Atributos mais importantes
# ============================
top_10_features = importances.tail(10).sort_values(ascending=False)
print("\n=== Top 10 Atributos mais importantes para previsão de vendas ===")
for i, (feature, importance) in enumerate(top_10_features.items(), 1):
    print(f"{i}. {feature}: {importance:.4f}")

top_10_features.to_csv(os.path.join(output_dir, "top10_feature_importance.csv"), header=["importance"])
with open(os.path.join(output_dir, "top10_feature_importance.txt"), "w", encoding="utf-8") as f:
    f.write("Top 10 Atributos mais importantes para previsão de vendas:\n")
    for i, (feature, importance) in enumerate(top_10_features.items(), 1):
        f.write(f"{i}. {feature}: {importance:.4f}\n")

# ============================
# 10. Resultado final em TXT (estilo usuário)
# ============================
result_txt_path = os.path.join(output_dir, "resultado_final.txt")
with open(result_txt_path, "w", encoding="utf-8") as f:
    f.write("Opção A:\n")
    f.write("- Objetivo: Prever quantos produtos serão comprados no mês\n")
    f.write("- Tipo de problema: Regressão\n")
    f.write("- Campo alvo: purchased_last_month\n")
    f.write("- Dificuldade: difícil\n")
    f.write("- Questão de pesquisa: Quais os atributos mais relevantes que impactam na quantidade de vendas de um produto? Existe correlação?\n\n")
    
    f.write("Avaliação do modelo:\n")
    f.write(f"- MAE  (Erro Absoluto Médio): {mae:.2f}\n")
    f.write(f"- RMSE (Raiz do Erro Quadrático Médio): {rmse:.2f}\n")
    f.write(f"- R²   (Coeficiente de Determinação): {r2:.2f}\n\n")
    
    f.write("Top 10 Atributos mais importantes para previsão de vendas:\n")
    for i, (feature, importance) in enumerate(top_10_features.items(), 1):
        f.write(f"{i}. {feature}: {importance:.4f}\n")

print(f"\nAnálise completa finalizada. Todos os arquivos foram salvos em: {output_dir}")
