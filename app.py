import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import joblib 
import pickle
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

# Set option to disable the warning
st.set_option('deprecation.showPyplotGlobalUse', False)

# Sidebar
import streamlit as st

# Set layout untuk sidebar di sebelah kanan
st.set_page_config(layout="wide")

# Sidebar
selected_category = st.sidebar.selectbox("Halaman", 
   ["1 - Dashboard", "Distribusi Data", "Analisis Perbandingan", "Analisis Hubungan",
   "Analisis Komposisi", "Data Predict"
])

# Data Handling (assuming data is in a CSV file)
URL = 'comet_goldfish.csv'
df = pd.read_csv(URL)

# Main Content
if selected_category == "1 - Dashboard":
    # Dashboard Section
    st.title("Memprediksi Umur Ikan Mas Komet Menggunakan Model Regresi untuk Optimalisasi Perawatan dan Pemeliharaan")
    st.image('https://m.media-amazon.com/images/I/71gwqv8aplL._AC_UF1000,1000_QL80_DpWeblab_.jpg', use_column_width=True)
    st.write(df)  # Menampilkan seluruh data pada halaman "Dashboard"
    st.write("Penjelasan Dashboard")

elif selected_category == "Distribusi Data":

    tabs = st.sidebar.radio('Pilihan Distribusi', ['Rentang Hidup Ikan Mas Komet', 'Panjang Rata-Rata Ikan Mas Komet', 'Habitat Ikan Mas Komet', 'Rata-Rata Berat Ikan Mas Komet'])

    if tabs == 'Rentang Hidup Ikan Mas Komet':
        # Judul halaman
        st.title('Distribusi Rentang Hidup Ikan Mas Komet')

        # Tampilkan histogram dengan seaborn
        plt.figure(figsize=(10, 6))
        sns.histplot(df['life_span'], kde=True, color='skyblue', bins=30)
        plt.title('Distribusi Rentang Hidup')
        plt.xlabel('Rentang Hidup')
        plt.ylabel('Frekuensi')

        # Tampilkan plot di Streamlit
        st.pyplot()

        st.write("Penjelasan Life Span")

    elif tabs == 'Panjang Rata-Rata Ikan Mas Komet':
        # Judul halaman
        st.title('Distribusi Panjang Rata-Rata Ikan Mas Komet dalam satuan inci')

        # Tampilkan histogram dengan seaborn di Streamlit
        plt.figure(figsize=(8, 6))
        sns.histplot(df['average_length(inches))'], bins=10, kde=True, color='skyblue')
        plt.title('Distribusi Panjang Rata-Rata')
        plt.xlabel('Panjang Rata-Rata')
        plt.ylabel('Frekuensi')

        # Tampilkan plot di Streamlit
        st.pyplot()

        st.write("Penjelasan Average Length")

    elif tabs == 'Habitat Ikan Mas Komet':
        # Judul halaman
        st.title('Distribusi Habitat Ikan Mas Komet')

        # Tampilkan bar plot dengan seaborn di Streamlit
        plt.figure(figsize=(8, 6))
        sns.countplot(x='habitat', data=df, palette='Set3')
        plt.title('Distribusi Habitat')
        plt.xlabel('Habitat')
        plt.ylabel('Frekuensi')

        # Tampilkan plot di Streamlit
        st.pyplot()

        st.write("Penjelasan Habitat")

    else:
        # Judul halaman
        st.title('Distribusi Berat Rata-Rata Ikan Mas Komet Berdasarkan Habitat')

        # Tampilkan boxplot dengan seaborn di Streamlit
        plt.figure(figsize=(10, 6))
        sns.boxplot(x='habitat', y='average_weight(inches))', data=df)
        plt.xlabel('Habitat')
        plt.ylabel('Rata-Rata Berat')
        plt.title('Distribusi Berat Rata-Rata')
        plt.xticks(rotation=45)

        # Tampilkan plot di Streamlit
        st.pyplot()

        st.write("Penjelasan Average Weight")



#ANALISIS PERBANDINGAN
elif selected_category == "Analisis Perbandingan":

    tabs = st.sidebar.radio('Pilihan Perbandingan', ['Panjang Rata-Rata dan Rentang Hidup', 'Berat Rata-Rata dan Rentang Hidup'])

    if tabs == 'Panjang Rata-Rata dan Rentang Hidup':
        # Judul halaman
        st.title('Analisis Perbandingan Panjang Rata-Rata dan Rentang Hidup Ikan Mas Komet')

        # Visualisasi regresi plot
        plt.figure(figsize=(8, 6))
        sns.regplot(x='average_length(inches))', y='life_span', data=df)
        plt.title('Regression Plot antara Average Length dan Life Span')
        plt.xlabel('Panjang Rata-Rata')
        plt.ylabel('Rentang Hidup')

        # Tampilkan plot menggunakan Streamlit
        st.pyplot(plt)

        st.write("Penjelasan Average Length dan Life Span")

    else:
        # Judul halaman
        st.title('Analisis Perbandingan Berat Rata-Rata dan Rentang Hidup Ikan Mas Komet')

        # Visualisasi scatter plot
        plt.figure(figsize=(8, 6))
        sns.scatterplot(x='average_weight(inches))', y='life_span', data=df, hue='habitat', palette='Set2')
        plt.title('Scatter Plot antara Average Weight dan Life Span')
        plt.xlabel('Berat Rata-Rata (inci)')
        plt.ylabel('Rentang Hidup')
        plt.legend(title='Habitat')

        # Tampilkan plot menggunakan Streamlit
        st.pyplot(plt)

        st.write("Penjelasan Average Weight dan Life Span")



#ANALISIS HUBUNGAN
elif selected_category == "Analisis Hubungan":
        
        # Judul halaman
        st.title('Analisis Hubungan Ikan Mas Komet')

        # Hapus kolom-kolom non-numerik
        df_numerik = df.select_dtypes(include=['number'])

        # Buat matriks korelasi
        matriks_korelasi = df_numerik.corr()

        # Visualisasi heatmap matriks korelasi
        plt.figure(figsize=(10, 8))
        sns.heatmap(matriks_korelasi, annot=True, cmap='coolwarm', fmt=".2f")
        plt.title('Matrix Plot dari Korelasi Antar Kolom')
        plt.xlabel('Kolom')
        plt.ylabel('Kolom')

        # Tampilkan plot menggunakan Streamlit
        st.pyplot(plt)

        st.write("Penjelasan")



#ANALISIS KOMPOSISI
elif selected_category == "Analisis Komposisi":

    tabs = st.sidebar.radio('Pilihan Komposisi', ['Jumlah Data Berdasarkan Habitat', 'Panjang Rata-Rata dan pH Air'])

    if tabs == 'Jumlah Data Berdasarkan Habitat':
        # Judul halaman
        st.title('Analisis Komposisi Jumlah Data Ikan Mas Komet Berdasarkan Habitat')

       # Hitung jumlah data berdasarkan habitat
        habitat_counts = df['habitat'].value_counts()

        # Visualisasi bar plot komposisi habitat
        plt.figure(figsize=(8, 6))
        habitat_counts.plot(kind='bar', color='skyblue')
        plt.title('Komposisi Habitat')
        plt.xlabel('Habitat')
        plt.ylabel('Jumlah Data')
        plt.xticks(rotation=45)

        # Tampilkan plot menggunakan Streamlit
        st.pyplot(plt)

        st.write("Penjelasan Jumlah Data Berdasarkan Habitat")

    else:
        # Judul halaman
        st.title('Analisis Komposisi Panjang Rata-Rata Ikan Mas Komet dan pH Air')

        # Visualisasi regresi plot
        plt.figure(figsize=(10, 6))
        sns.regplot(x='average_length(inches))', y='ph_of_water', data=df)
        plt.title('Regression Plot antara Average Length dan pH of Water')
        plt.xlabel('Panjang Rata-Rata')
        plt.ylabel('pH Air')

        # Tampilkan plot menggunakan Streamlit
        st.pyplot(plt)

        st.write("Panjang Rata-Rata dan pH Air")



else:
    # Load the pre-trained linear regression model
    with open("modell.pkl", "rb") as f:
        model = pickle.load(f)

    # Function to predict life span based on input features
    def predict_life_span(ph_of_water, life_span):
        # Create a DataFrame from input data
        input_data = pd.DataFrame({
            'ph_of_water': [ph_of_water],
            'life_span': [life_span]
        })

        # Make prediction
        predicted_life_span = model.predict(input_data)
        return predicted_life_span[0]  # Return the predicted life span value

    # Streamlit app
    def main():
        # Sidebar title
        st.title("Input Features")

        # Input field for pH of water
        ph_of_water = st.slider("pH of Water", min_value=0.1, max_value=14.0, value=7.0, step=0.1)
        life_span = st.slider("Life Span", min_value=0.1, max_value=50.0, value=10.0, step=0.1)

        # Make prediction when the user clicks the button
        if st.button("Predict Life Span"):
            predicted_life_span = predict_life_span(ph_of_water, life_span)
            st.write(f"Predicted Life Span: {predicted_life_span:.2f} years")

    # Run the app
    if __name__ == "__main__":
        main()
