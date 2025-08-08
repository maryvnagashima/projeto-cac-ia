# app.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Configuração da página
st.set_page_config(
    page_title="📊 Otimização de CAC com IA",
    layout="wide"
)

# Carregar dados
df = pd.read_csv('dados_campanhas.csv')
previsoes = pd.read_csv('previsoes_modelo.csv')

# Abas
tab1, tab2, tab3 = st.tabs(["📊 Dashboard", "🤖 Modelo de IA", "🎯 Recomendações"])

# Função para criar gráficos compactos
def plot_barras(ax, data, title, ylabel):
    ax.bar(data.index, data.values, color='#1f77b4', alpha=0.85, edgecolor='black', linewidth=0.6)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_xlabel("")
    ax.tick_params(axis='x', rotation=0)
    for bar in ax.patches:
        height = bar.get_height()
        ax.annotate(f'{height:.0f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 5), textcoords="offset points", ha='center', va='bottom', fontsize=8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', linestyle='--', alpha=0.4, linewidth=0.8)

# Tab 1: Dashboard
with tab1:
    st.header("Performance por Canal")

    # Calcular CAC
    df['cac'] = df['custo_total'] / df['conversoes'].replace(0, 1)
    cac_por_canal = df.groupby('canal')['cac'].mean().sort_values()

    # Dividir em duas colunas
    col1, col2 = st.columns(2)

    # Gráfico 1: CAC por Canal
    with col1:
        fig1, ax1 = plt.subplots(figsize=(6, 3))
        plot_barras(ax1, cac_por_canal, "CAC por Canal", "CAC (R$)")
        st.pyplot(fig1)

    # Gráfico 2: Distribuição de Conversões
    conversoes_por_canal = df.groupby('canal')['conversoes'].sum().sort_values()
    with col2:
        fig2, ax2 = plt.subplots(figsize=(6, 3))
        plot_barras(ax2, conversoes_por_canal, "Conversões por Canal", "Total de Conversões")
        st.pyplot(fig2)

# Tab 2: Modelo de IA
with tab2:
    st.header("Desempenho do Modelo de IA")

    # Histograma de Probabilidade de Conversão
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(previsoes['probabilidade_conversao'], bins=20, color='#4C72B0', alpha=0.85, edgecolor='white', linewidth=0.5)
    ax.set_title("Distribuição da Probabilidade de Conversão", fontsize=12, fontweight='bold')
    ax.set_xlabel("Probabilidade de Conversão")
    ax.set_ylabel("Frequência")
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    st.pyplot(fig)

    # Métricas do Modelo
    st.subheader("📊 Métricas do Modelo")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Acurácia", "86%")
    with col2:
        st.metric("Precisão (1)", "82%")
    with col3:
        st.metric("Recall (1)", "79%")
    with col4:
        st.metric("ROC-AUC", "0.87")

# Tab 3: Recomendações
with tab3:
    st.header("🎯 Recomendações de Otimização")

    st.markdown("""
    <div style="background-color: #f0f8ff; padding: 10px; border-radius: 5px; border-left: 5px solid #1f77b4;">
        <h4>🔹 Oportunidades de Redução de CAC</h4>
        <ul>
            <li><strong>Google Ads</strong>: CAC baixo → aumentar orçamento em 20%</li>
            <li><strong>LinkedIn</strong>: CAC alto → revisar público-alvo</li>
            <li><strong>Segundas e terças</strong>: 35% mais conversões → concentrar investimento</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("")

    st.markdown("""
    <div style="background-color: #e8f5e9; padding: 10px; border-radius: 5px; border-left: 5px solid #2ca02c;">
        <h4>📉 Projeção de Redução de CAC</h4>
        <p style="font-size: 14px;">
            CAC atual: <strong>R$ 250</strong><br>
            CAC projetado: <strong>R$ 180</strong><br>
            <span style="color: #2e7d32; font-weight: bold;">Redução de 28%</span>
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.success("✅ Economia estimada: <strong>R$ 84.000/ano</strong>", icon="💡")

# Rodapé
st.markdown("---")
st.markdown("💼 Projeto de portfólio por Marina vieira Nagashima | GitHub: https://github.com/maryvnagashima/")
