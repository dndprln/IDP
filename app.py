import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import joblib 
import pickle
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
   ["1 - Dashboard", "2 - Distribusi Data", "3 - Analisis Perbandingan", "4 - Analisis Hubungan",
   "5 - Analisis Komposisi", "6 - Data Predict"
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
    st.write("Ikan mas komet memiliki ciri-ciri yang khas seperti bentuk ekornya yang menyerupai komet, tubuhnya yang bulat dan pipih, serta warnanya yang cerah seperti oranye, merah, atau kuning dengan bintik-bintik hitam di sekitarnya. Ikan ini populer di dunia akuarium karena keindahan warna dan gerakannya yang anggun. Mereka juga dikenal sebagai ikan yang ramah dan mudah dijaga dengan baik asalkan mendapat perawatan yang sesuai dan lingkungan yang cocok.")
    st.write("Tujuan bisnis adalah memprediksi umur ikan mas komet dengan akurat berdasarkan atribut-atribut yang tersedia untuk membantu optimalisasi perawatan dan pemeliharaan ikan tersebut.")
    st.write("Data dengan judul Predict lifespan of a comet goldfish berasal dari Kaggle.", 
             "Link: https://www.kaggle.com/datasets/stealthtechnologies/predict-lifespan-of-a-comet-goldfish/code")



# DISTRIBUSI DATA
elif selected_category == "2 - Distribusi Data":

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

        st.write('Grafik ini menunjukkan bahwa rata-rata ikan mas komet hidup selama 10 tahun. Namun, ada juga beberapa ikan mas komet yang memiliki rentang hidup yang jauh lebih pendek atau lebih panjang dari rata-rata. Hal ini dapat disebabkan oleh berbagai faktor, seperti kesehatan, genetik, dan kualitas air.')

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

        st.write("Grafik ini menunjukkan bahwa rata-rata ikan mas komet memiliki panjang 5 cm. Namun, ada juga beberapa ikan mas komet yang memiliki panjang yang jauh lebih pendek atau lebih panjang dari rata-rata. Hal ini dapat disebabkan oleh berbagai faktor, seperti genetik, nutrisi, dan lingkungan.")

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

        st.write("Grafik ini menunjukkan bahwa ikan mas komet paling banyak ditemukan di Lakes (2), lalu di Ponds(0), rivers(3), slowmovingwaters(4), dan idlewater(1)")

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

        st.write("Berdasarkan analisis Box plot, dapat disimpulkan bahwa Habitat 3 (Rivers) memiliki berat rata-rata tertinggi untuk ikan mas komet, diikuti oleh Habitat 2 (lakes) dan Habitat 1 (idlewater). Habitat 3 (Rivers) juga menunjukkan variabilitas tertinggi dalam distribusi berat, sedangkan Habitat 1 (idlewater) memiliki variabilitas terendah. Habitat 2 (lakes) memiliki berat rata-rata yang lebih dekat dengan berat median keseluruhan ikan mas komet di ketiga habitat.")




#ANALISIS PERBANDINGAN
elif selected_category == "3 - Analisis Perbandingan":

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

        st.write("Grafik menunjukkan plot sebar dengan garis regresi, berjudul Regression Plot antara Panjang Rata-Rata dan Rentang Hidup yang menunjukkan bahwa data dan label dalam bahasa Indonesia. Sumbu x dilabeli Panjang Rata-Rata dan berkisar dari 0 hingga 20. Sumbu y dilabeli Rentang Hidup dan berkisar dari 0 hingga sekitar 30. Titik data digambarkan sebagai titik biru tersebar di seluruh grafik, dengan konsentrasi titik di antara bagian tengah dan bagian bawah rentang sumbu y. Garis regresi biru digambar secara horizontal melintasi plot, menunjukkan sedikit atau tidak ada kemiringan, yang menandakan hubungan linear yang lemah atau tidak ada antara variabel Panjang Rata-Rata dan Rentang Hidup. Hal ini mengimplikasikan bahwa perubahan dalam panjang rata-rata tidak secara konsisten memprediksi perubahan dalam rentang hidup berdasarkan data yang disajikan dalam plot sebar ini.")

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

        st.write("Plot ini berisi banyak titik berwarna yang masing-masing mewakili titik data dengan warna yang berbeda sesuai dengan habitat yang berbeda, seperti yang ditunjukkan oleh legenda di sudut kanan atas. Legenda menunjukkan Habitat diikuti oleh angka 0 hingga 4, masing-masing terkait dengan warna yang berbeda. Habitat yang tepat yang diacu oleh angka-angka ini tidak spesifik dalam gambar. Plot sebar ini tidak menunjukkan tren atau korelasi yang jelas antara berat rata-rata dan rentang hidup, karena titik-titik tersebar luas di seluruh area plotting. Penggunaan berbagai warna menunjukkan bahwa titik data dikategorikan berdasarkan habitat, namun tanpa konteks tambahan, sulit untuk menarik kesimpulan spesifik tentang hubungan antara variabel tersebut atau dampak habitat pada variabel tersebut.")



#ANALISIS HUBUNGAN
elif selected_category == "4 - Analisis Hubungan":
        
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

        st.write("Gambar menampilkan plot matriks dengan judul Matrix Plot dari Korelasi Antar Kolom, yang merupakan representasi visual dari koefisien korelasi antara variabel-variabel yang berbeda dalam sebuah dataset")
        st.write("Koefisien korelasi berkisar dari -1 hingga 1, di mana 1 menunjukkan korelasi positif sempurna, -1 menunjukkan korelasi negatif sempurna, dan 0 menunjukkan tidak adanya korelasi. Warna dalam matriks ini berkisar dari biru tua (korelasi negatif kuat) hingga merah tua (korelasi positif kuat), dengan warna-warna lebih terang menunjukkan korelasi yang lebih lemah. Beberapa korelasi yang mencolok yang terlihat dalam matriks ini antara lain:")
        st.write('1. Korelasi negatif kuat (biru tua) sebesar -0,62 antara average_length (inci) dan weight_length_ratio.')
        st.write('2. Korelasi positif kuat (merah tua) sebesar 0,97 antara life_span dan age_group.')



#ANALISIS KOMPOSISI
elif selected_category == "5 - Analisis Komposisi":

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

    st.write("1. Habitat 3 (Rivers): Habitat 3 (Rivers) memiliki jumlah titik data (ikan mas komet) terbanyak yang tercatat, yaitu sekitar 60% dari total data.")
    st.write('2. Habitat 2 (lakes): Habitat 2 (lakes) memiliki jumlah titik data (ikan mas komet) tertinggi kedua yang tercatat, yaitu sekitar 40% dari total data.')
    st.write("3. Habitat 1 (idlewater): Habitat 1 (idlewater) memiliki jumlah titik data (ikan mas komet)  terendah yang tercatat, yaitu sekitar 20% dari total data.")



# DATA PREDIKSI
else:
    # Load the pre-trained linear regression model
    with open("modell.pkl", "rb") as f:
        model = pickle.load(f)

    # Function to predict life span based on input features
    def predict_life_span(habitat, ph_of_water):
        # Create a DataFrame from input data
        input_data = pd.DataFrame({
            'habitat': [habitat],
            'ph_of_water': [ph_of_water]
        })

        # Make prediction
        predicted_life_span = model.predict(input_data)
        return predicted_life_span[0]  # Return the predicted life span value

    # Streamlit app
    def main():
        # Sidebar title
        st.title("Masukkan Nilai yang ingin diprediksi")

        # Input field for pH of water
        habitat = st.slider("Habitat", min_value=0, max_value=4, value=2, step=1)  # Assuming habitat values from 0 to 4
        ph_of_water = st.slider("pH of Water", min_value=0.1, max_value=14.0, value=7.0, step=0.1)

        # Make prediction when the user clicks the button
        if st.button("Predict"):
            predicted_life_span = predict_life_span(habitat, ph_of_water)
            st.write(f"Prediksi Rentang Hidup Ikan Mas Komet yaitu: {predicted_life_span:.2f} years")

    # Run the app
    if __name__ == "__main__":
        main()
