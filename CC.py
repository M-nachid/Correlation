import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['font.family'] = 'Times New Roman'  # Set font

st.title("Cross-Quantile Correlation Heatmap")

# Sidebar for quantile settings
st.sidebar.header("Settings")
quantile_steps = st.sidebar.selectbox("Quantile Groups", 4, index=0)

# Upload CSV file
uploaded_file = st.file_uploader("Upload your XLS file", type=["xlsx"])

if uploaded_file is not None:
    df = pd.read_excel(uploaded_file)
    
    # Filter only numeric columns
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    
    if len(numeric_cols) < 2:
        st.warning("The uploaded dataset must contain at least two numeric columns.")
    else:
        # Let user choose two variables
        var1 = st.selectbox("Select Variable 1", numeric_cols, index=0)
        var2 = st.selectbox("Select Variable 2", numeric_cols, index=1)
        
        data1 = df[var1].dropna()
        data2 = df[var2].dropna()


        # Compute quantiles
        #probs = np.linspace(0, 1, quantile_steps + 1)
        quantiles1 = np.quantile(data1, [0, 0.25, 0.5, 0.75, 1])
        quantiles2 = np.quantile(data2, [0, 0.25, 0.5, 0.75, 1])

        # Cross-quantile correlation
        corr_matrix = np.full((4, 4), np.nan)
        for i in range(4):
            for j in range(4):
                idx1 = (data1 >= quantiles1[i]) & (data1 < quantiles1[i+1])
                idx2 = (data2 >= quantiles2[j]) & (data2 < quantiles2[j+1])
                common_idx = idx1 & idx2
                if np.sum(common_idx) > 1:
                    corr_matrix[i, j] = np.corrcoef(data1[common_idx], data2[common_idx])[0, 1]

        labels = [f"Q{i+1}" for i in range(4)]
        df_corr = pd.DataFrame(corr_matrix, index=labels, columns=labels)

        # Display correlation matrix
        st.subheader("Cross-Quantile Correlation Matrix")
        st.dataframe(df_corr.style.background_gradient(cmap='coolwarm', axis=None).format("{:.3f}"))

        # Plot heatmap
        st.subheader("Heatmap")
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(df_corr, annot=True, cmap="coolwarm", vmin=-1, vmax=1, linecolor='lightblue', linewidths=2,
                    xticklabels=labels, yticklabels=labels, fmt=".3f", ax=ax)
        ax.set_xlabel(f"{var1} Quantiles")
        ax.set_ylabel(f"{var2} Quantiles")
        plt.xticks(rotation= 45)
        plt.yticks(rotation= 45)
        ax.set_title("Cross-Quantile Correlation Heatmap")
        st.pyplot(fig)
else:
    st.info("Upload a EXCEL file to begin.")
