import streamlit as st
import pandas as pd
import numpy as np

st.title('Streamlit 安裝成功 🎉')
st.header('這是一個在 Conda 環境中運行的應用程式')

# 創建一個互動式滑桿
value = st.slider('選擇一個值:', 0, 100, 50)
st.write(f'當前選擇的值是: {value}')

# 顯示一個簡單的數據表格
df = pd.DataFrame(
    np.random.randn(10, 2),
    columns=['A', 'B']
)
st.dataframe(df)