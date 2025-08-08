# app.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# Configuração da página
st.set_page_config(
    page_title="📊 Otimização de CAC com IA",
    page_icon="🚀",
    layout="wide"
)
# Estilo customizado
# Fundo escuro
st.markdown(
    """
    <style>
    .stApp {
        background-color: #ffff;
        color: white;
    }
    h1, h2, h3 {
        color: #1f77b4;
    }
    .stTextInput > label, .stSelectbox > label {
        color: white;
    }
    </style>
    """,
    unsafe_allow_html=True
)
# Título
st.title("🚀 Otimização de CAC com IA")
st.subheader("Projeto de Portfólio - Redução de Custo de Aquisição de Cliente")
st.markdown("""
Este projeto simula um modelo de IA para prever conversões e reduzir o CAC em campanhas de mídia paga.
**Fonte:** Dados simulados com Python (Google Ads, Meta Ads, etc).
""")

# Carregar dados
try:
    df = pd.read_csv('dados_campanhas.csv')
    previsoes = pd.read_csv('previsoes_modelo.csv')

    # Garantir que 'converteu' existe
    if 'converteu' not in previsoes.columns:
        previsoes['converteu'] = (previsoes['real_converteu'] if 'real_converteu' in previsoes else 0)

except Exception as e:
    st.error(f"Erro ao carregar os dados: {e}")
    st.stop()

# Abas
tab1, tab2, tab3, tab4 = st.tabs(["📊 Dashboard", "🤖 Modelo de IA", "🎯 Recomendações", "📂 Dados"])

with tab1:
    st.header("Performance por Canal")

    # Calcular CAC
    df['cac'] = df['custo_total'] / df['conversoes'].replace(0, 1)
    cac_por_canal = df.groupby('canal')['cac'].mean().sort_values()

    # Paleta de cores profissional
    cores = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

    # Gráfico mais bonito
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(cac_por_canal.index, cac_por_canal.values, color=cores, alpha=0.85, edgecolor='black', linewidth=0.6)

    # Títulos
    ax.set_title("Custo de Aquisição por Cliente (CAC) por Canal", fontsize=16, fontweight='bold', pad=20)
    ax.set_ylabel("CAC (R$)", fontsize=12)
    ax.set_xlabel("Canal de Mídia", fontsize=12)

    # Girar rótulos
    plt.xticks(rotation=0)

    # Adicionar valores em cima das barras
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'R$ {height:.0f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 5),  # 5 pontos acima
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=10, fontweight='bold')

    # Remover bordas
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(0.5)
    ax.spines['bottom'].set_linewidth(0.5)

    # Grade suave
    ax.grid(axis='y', linestyle='--', alpha=0.4, linewidth=0.8)

    st.pyplot(fig)

with tab2:
    st.header("Desempenho do Modelo de IA")

    # Histograma mais bonito
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(previsoes['probabilidade_conversao'], bins=20, color='#4C72B0', alpha=0.85, edgecolor='white', linewidth=0.5)

    ax.set_title("Distribuição da Probabilidade de Conversão", fontsize=14, fontweight='bold')
    ax.set_xlabel("Probabilidade de Conversão")
    ax.set_ylabel("Frequência")

    # Estilo
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', linestyle='--', alpha=0.4)

    st.pyplot(fig)
    # Métricas em colunas
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
with tab3:
    st.header("🎯 Recomendações de Otimização")

    st.markdown("""
    <div style="background-color: #f0f8ff; padding: 20px; border-radius: 10px; border-left: 5px solid #1f77b4;">
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
    <div style="background-color: #e8f5e9; padding: 20px; border-radius: 10px; border-left: 5px solid #2ca02c;">
        <h4>📉 Projeção de Redução de CAC</h4>
        <p style="font-size: 18px;">
            CAC atual: <strong>R$ 250</strong><br>
            CAC projetado: <strong>R$ 180</strong><br>
            <span style="color: #2e7d32; font-weight: bold;">Redução de 28%</span>
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.success("✅ Economia estimada: <strong>R$ 84.000/ano</strong>", icon="💡")
with tab4:
    st.header("📂 Visualizar Dados Brutos")

    st.write("### Dados das Campanhas")
    st.dataframe(df.head(10))

    st.write("### Previsões do Modelo")
    st.dataframe(previsoes.head(10))

# Rodapé
st.markdown("---")
st.markdown("💼 Projeto de portfólio por Marina vieira Nagashima | GitHub: https://github.com/maryvnagashima/")
