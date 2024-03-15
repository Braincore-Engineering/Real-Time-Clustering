import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from kmodes.kprototypes import KPrototypes


def modelData(data):
    df = pd.read_csv(data, delimiter="\t")  # Menggunakan data terbaru dari file
    df_std = df[["Umur", "NilaiBelanjaSetahun"]]
    df_stdx = StandardScaler().fit_transform(df_std)
    stdx = pd.DataFrame(df_stdx, columns=['Umur', 'NilaiBelanjaSetahun'])
    kolom_kategorikal = ['Jenis Kelamin', 'Profesi', 'Tipe Residen']
    df_encode = df[kolom_kategorikal].apply(LabelEncoder().fit_transform)
    
    # Reset indeks DataFrame
    df.reset_index(drop=True, inplace=True)
    
    df_model = pd.concat([df_encode, stdx], axis=1)
    model = KPrototypes(n_clusters=5, random_state=75)
    cluster_prediction = model.fit_predict(df_model, categorical=[0,1,2])
    df["clusters"] = cluster_prediction
    df_final = df.copy()
    df_final["clusters"] = cluster_prediction
    
    df_final['segmen'] = df_final['clusters'].map({
        0: 'Diamond Young Member',
        1: 'Diamond Senior Member',
        2: 'Silver Member',
        3: 'Gold Young Member',
        4: 'Gold Senior Member'
    })
    
    return df_final
