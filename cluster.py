import pandas as pd
from flask import Flask, request, render_template, flash, redirect, url_for
from flask_socketio import SocketIO, emit
from sklearn.preprocessing import LabelEncoder, StandardScaler
from kmodes.kprototypes import KPrototypes
from model import modelData

app = Flask(__name__)
app.config['SECRET_KEY'] = 'Secret'
socketio = SocketIO(app)

# Load the customer segments data
df = pd.read_csv('customer_segments.txt', delimiter='\t')

# Your clustering model implementation goes here

# Fungsi generate_customer_id
def generate_customer_id():
    num_existing_data = len(df)
    return f'CUST-{num_existing_data + 1}'

# Menambahkan generate_customer_id ke dalam konteks Flask
@app.context_processor
def utility_processor():
    return dict(generate_customer_id=generate_customer_id)

@app.route('/')
def index():
    clustering_result = modelData('customer_segments.txt')

    return render_template('index.html', clustering_result=clustering_result)
    # return render_template('index.html', data=df)

@app.route('/add_customer', methods=['POST'])
def add_customer():
    global df
    if request.method == 'POST':
        # Ambil data dari form
        new_data = {
            'Customer_ID': generate_customer_id(),
            'Nama Pelanggan': request.form['nama_pelanggan'],
            'Jenis Kelamin': request.form['jenis_kelamin'],
            'Umur': int(request.form['umur']),
            'Profesi': request.form['profesi'],
            'Tipe Residen': request.form['tipe_residen'],
            'NilaiBelanjaSetahun': int(request.form['nilai_belanja']),
        }
        
        new_data = pd.DataFrame([new_data])
        df = pd.concat([df, new_data], ignore_index=True)

        # Tambahkan data baru ke DataFrame
        # df = df.append(new_data, ignore_index=True)

        # Simpan DataFrame yang baru ke dalam file
        df.to_csv('customer_segments.txt', sep='\t', index=False)

        # Menggunakan modelData dari file model.py
        clustering_result = modelData('customer_segments.txt')

        # Emit event ke semua klien dengan hasil clustering
        socketio.emit('update_clusters', clustering_result.to_json())

        flash('Data customer baru berhasil ditambahkan!', 'success')

    return redirect(url_for('index'))


if __name__ == '__main__':
    socketio.run(app, debug=True)
