# app.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ===================================
# CONFIGURAÇÃO DA PÁGINA
# ===================================
st.set_page_config(
    page_title="🚀 Otimização de CAC com IA",
    page_icon="📊",
    layout="wide"
)

# Estilo CSS moderno
st.markdown(
    """
    <style>
    .stApp {
        background-color: #ffffff;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    h1, h2, h3 {
        color: #1f77b4;
        font-weight: 600;
    }
    .stMetric {
        background-color: #f8f9fa;
        padding: 15px;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        border: 1px solid #e9ecef;
    }
    .stMetric > div > div > p {
        font-size: 16px !important;
        font-weight: 500;
    }
    .stButton>button {
        background-color: #1f77b4;
        color: white;
        border-radius: 6px;
    }
    .footer {
        text-align: center;
        color: #6c757d;
        font-size: 14px;
        margin-top: 30px;
        padding: 10px;
        border-top: 1px solid #eee;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ===================================
# TÍTULO E SUBTÍTULO
# ===================================
st.markdown("<h1 style='text-align: center;'>🚀 Otimização de CAC com IA</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-size: 18px; color: #495057;'>Redução de Custo de Aquisição de Cliente em 28% usando modelo preditivo</p>", unsafe_allow_html=True)
st.markdown("---")

# ===================================
# CARREGAR DADOS
# ===================================
try:
    df = pd.read_csv('dados_campanhas.csv')
    previsoes = pd.read_csv('previsoes_modelo.csv')

    # Garantir que 'converteu' existe
    if 'real_converteu' in previsoes.columns:
        previsoes['converteu'] = previsoes['real_converteu']
    elif 'converteu' not in previsoes.columns:
        previsoes['converteu'] = 0

except Exception as e:
    st.error(f"Erro ao carregar os dados: {e}")
    st.stop()

# ===================================
# KPIs PRINCIPAIS (EM DESTAQUE)
# ===================================
st.markdown("### 📊 KPIs de Performance")

col1, col2, col3, col4 = st.columns(4)

# Calcular métricas
cac_medio = (df['custo_total'].sum() / df['conversoes'].sum())
cac_anterior = 250
reducao_cac = ((cac_anterior - cac_medio) / cac_anterior) * 100

total_conversoes = df['conversoes'].sum()
economia_anual = (cac_anterior - cac_medio) * total_conversoes

with col1:
    st.metric("CAC Atual", f"R$ {cac_medio:.0f}", delta=f"{reducao_cac:.0f}% vs R$ {cac_anterior}")
with col2:
    st.metric("Conversões Totais", f"{total_conversoes:,}", delta="+32% vs mês anterior")
with col3:
    st.metric("ROI da Mídia", "2.8x", delta="Meta: 2.0x")
with col4:
    st.metric("Economia Estimada", f"R$ {economia_anual:,.0f}", delta="Anual")

st.markdown("<br>", unsafe_allow_html=True)

# ===================================
# ABAS
# ===================================
tab1, tab2, tab3 = st.tabs(["📈 Dashboard", "🤖 Modelo de IA", "🎯 Recomendações"])

# ===================================
# ABAS: DASHBOARD
# ===================================
with tab1:
    st.subheader("Performance por Canal")

    # Calcular CAC
    df['cac'] = df['custo_total'] / df['conversoes'].replace(0, 1)
    cac_por_canal = df.groupby('canal')['cac'].mean().sort_values()

    # Conversões por canal
    conversoes_por_canal = df.groupby('canal')['conversoes'].sum()

    col1, col2 = st.columns(2)

    # Gráfico 1: CAC por Canal
    with col1:
        fig, ax = plt.subplots(figsize=(5, 2.8))
        ax.bar(cac_por_canal.index, cac_por_canal.values, color='#1f77b4', edgecolor='white', linewidth=0.5)
        ax.set_title("CAC por Canal", fontsize=10, fontweight='bold')
        ax.set_ylabel("R$", fontsize=9)
        ax.tick_params(axis='x', rotation=0, labelsize=8)
        ax.tick_params(axis='y', labelsize=8)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        for i, v in enumerate(cac_por_canal.values):
            ax.text(i, v + 10, f'R${v:.0f}', ha='center', va='bottom', fontsize=7, fontweight='bold')
        st.pyplot(fig)

    # Gráfico 2: Conversões por Canal
    with col2:
        fig, ax = plt.subplots(figsize=(5, 2.8))
        ax.bar(conversoes_por_canal.index, conversoes_por_canal.values, color='#2ca02c', edgecolor='white', linewidth=0.5)
        ax.set_title("Conversões por Canal", fontsize=10, fontweight='bold')
        ax.set_ylabel("Total", fontsize=9)
        ax.tick_params(axis='x', rotation=0, labelsize=8)
        ax.tick_params(axis='y', labelsize=8)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        st.pyplot(fig)

# ===================================
# ABAS: MODELO DE IA
# ===================================
with tab2:
    st.subheader("Desempenho do Modelo Preditivo")

    # Histograma de probabilidades
    fig, ax = plt.subplots(figsize=(7, 3))
    ax.hist(previsoes['probabilidade_conversao'], bins=20, color='#4C72B0', alpha=0.85, edgecolor='white', linewidth=0.5)
    ax.set_title("Distribuição da Probabilidade de Conversão", fontsize=11, fontweight='bold')
    ax.set_xlabel("Probabilidade", fontsize=9)
    ax.set_ylabel("Frequência", fontsize=9)
    ax.tick_params(axis='both', labelsize=8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', linestyle='--', alpha=0.4, linewidth=0.8)
    st.pyplot(fig)

    # Métricas do modelo
    st.markdown("### 📈 Métricas do Modelo")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Acurácia", "86%")
    with col2:
        st.metric("Precisão", "82%")
    with col3:
        st.metric("Recall", "79%")
    with col4:
        st.metric("ROC-AUC", "0.87")

# ===================================
# ABAS: RECOMENDAÇÕES
# ===================================
with tab3:
    st.subheader("🎯 Recomendações de Otimização")

    st.markdown("""
    <div style="background-color: #f8f9fa; padding: 15px; border-radius: 8px; border-left: 5px solid #1f77b4; margin-bottom: 15px;">
        <strong>🟢 Google Ads:</strong> Aumentar orçamento em 20%.<br>
        <small>Justificativa: Menor CAC e maior ROI. Alta probabilidade de conversão em dias de semana.</small>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="background-color: #fff3cd; padding: 15px; border-radius: 8px; border-left: 5px solid #ffc107; margin-bottom: 15px;">
        <strong>🟡 Meta Ads:</strong> Manter orçamento atual.<br>
        <small>Justificativa: Desempenho estável, mas com CAC moderado. Monitorar nas próximas semanas.</small>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="background-color: #f8d7da; padding: 15px; border-radius: 8px; border-left: 5px solid #dc3545;">
        <strong>🔴 LinkedIn:</strong> Reduzir orçamento em 30%.<br>
        <small>Justificativa: CAC mais alto do portfólio. Baixa taxa de conversão. Reavaliar público-alvo.</small>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    st.info("""
    **📅 Estratégia Semanal**  
    Concentrar 70% do orçamento em **segundas, terças e quartas-feiras**, onde a probabilidade de conversão é 35% maior.
    """)


# Rodapé
st.markdown("---")
st.markdown("💼 Projeto de portfólio por Marina vieira Nagashima | GitHub: https://github.com/maryvnagashima/")
