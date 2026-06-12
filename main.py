# Algoritma K-Means dan Decision Tree untuk Prediksi Penerimaan
# Calon Mahasiswa Pascasarjana pada Universitas Indonesia
# Referensi: Hutagaol, R., Ardiansyah, B., Daulay, I., & Rahmaddeni (2022)

import os
import sys
import warnings
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    silhouette_score
)
from sklearn.preprocessing import LabelEncoder, StandardScaler

warnings.filterwarnings("ignore")

FILE_DATASET = "dataset_SDA.xlsx"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FILE_PATH = os.path.join(SCRIPT_DIR, FILE_DATASET)


def load_dataset():
    if not os.path.exists(FILE_PATH):
        print(f"File '{FILE_DATASET}' tidak ditemukan di: {SCRIPT_DIR}")
        sys.exit(1)
    df = pd.read_excel(FILE_PATH)
    print(f"Dataset dimuat: {len(df)} baris, {len(df.columns)} kolom")
    return df


# Menu 1 - Tampilkan Dataset
def tampilkan_dataset(df):
    print("\n--- TAMPILKAN DATASET ---\n")
    print(f"Jumlah Data  : {len(df)} baris")
    print(f"Jumlah Kolom : {len(df.columns)} kolom")
    print(f"Kolom        : {', '.join(df.columns.tolist())}")

    print("\n5 Data Pertama:")
    print(df.head().to_string(index=False))

    print("\n5 Data Terakhir:")
    print(df.tail().to_string(index=False))

    print("\nStatistik Deskriptif:")
    print(df.describe().to_string())

    print("\nDistribusi Label (Graduated):")
    for label, count in df["Graduated"].value_counts().items():
        persen = count / len(df) * 100
        print(f"  {label} : {count} ({persen:.1f}%)")

    missing = df.isnull().sum()
    if missing.sum() == 0:
        print("\nTidak ada missing values.")
    else:
        print("\nMissing values:")
        print(missing[missing > 0].to_string())


# Menu 2 - Decision Tree
def decision_tree_analysis(df):
    print("\n--- ANALISIS DECISION TREE ---\n")

    le = LabelEncoder()
    df_encoded = df.copy()
    df_encoded["Graduated"] = le.fit_transform(df_encoded["Graduated"])
    label_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
    print(f"Label Encoding: {label_mapping}")

    feature_cols = ["GRE Score", "TOEFL Score", "CGPA", "Research"]
    X = df_encoded[feature_cols]
    y = df_encoded["Graduated"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    print(f"Data Training : {len(X_train)} baris")
    print(f"Data Testing  : {len(X_test)} baris")

    dt_model = DecisionTreeClassifier(
        criterion="entropy",
        max_depth=5,
        random_state=42
    )
    dt_model.fit(X_train, y_train)
    print("Model berhasil ditraining.")

    y_pred = dt_model.predict(X_test)

    print("\nHasil Prediksi pada Data Testing:")
    hasil_test = X_test.copy()
    hasil_test["Actual"] = le.inverse_transform(y_test)
    hasil_test["Predicted"] = le.inverse_transform(y_pred)
    hasil_test["Status"] = ["Benar" if a == p else "Salah"
                            for a, p in zip(hasil_test["Actual"], hasil_test["Predicted"])]
    print(hasil_test.to_string(index=False))

    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    cm_df = pd.DataFrame(
        cm,
        index=[f"Aktual: {c}" for c in le.classes_],
        columns=[f"Prediksi: {c}" for c in le.classes_]
    )
    print(cm_df.to_string())

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    akurasi = accuracy_score(y_test, y_pred) * 100
    print(f"Akurasi Decision Tree: {akurasi:.2f}%")

    print("\nAturan (Rules) Decision Tree:")
    rules = export_text(dt_model, feature_names=feature_cols, decimals=2)
    print(rules)

    print("Feature Importance:")
    importances = dt_model.feature_importances_
    fi_df = pd.DataFrame({
        "Fitur": feature_cols,
        "Importance": importances
    }).sort_values("Importance", ascending=False)
    for _, row in fi_df.iterrows():
        print(f"  {row['Fitur']:15s} : {row['Importance']:.4f}")

    return dt_model, le, feature_cols


# Menu 3 - Visualisasi Decision Tree
def visualisasi_decision_tree(df):
    print("\n--- VISUALISASI DECISION TREE ---\n")

    try:
        import matplotlib
        matplotlib.use("TkAgg")
        import matplotlib.pyplot as plt
        from sklearn.tree import plot_tree
    except ImportError:
        print("matplotlib belum terinstall. Install dengan: pip install matplotlib")
        return

    le = LabelEncoder()
    df_encoded = df.copy()
    df_encoded["Graduated"] = le.fit_transform(df_encoded["Graduated"])

    feature_cols = ["GRE Score", "TOEFL Score", "CGPA", "Research"]
    X = df_encoded[feature_cols]
    y = df_encoded["Graduated"]

    X_train, _, y_train, _ = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    dt_model = DecisionTreeClassifier(criterion="entropy", max_depth=5, random_state=42)
    dt_model.fit(X_train, y_train)

    print("Membuka jendela grafik...")

    fig, ax = plt.subplots(figsize=(20, 10))
    plot_tree(
        dt_model,
        feature_names=feature_cols,
        class_names=le.classes_,
        filled=True,
        rounded=True,
        fontsize=10,
        ax=ax,
        impurity=True
    )
    ax.set_title("Decision Tree - Prediksi Penerimaan Mahasiswa Pascasarjana",
                 fontsize=16, fontweight="bold", pad=20)
    plt.tight_layout()

    save_path = os.path.join(SCRIPT_DIR, "decision_tree_visual.png")
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Gambar disimpan: {save_path}")
    plt.show()


# Menu 4 - K-Means
def kmeans_analysis(df):
    print("\n--- ANALISIS K-MEANS CLUSTERING ---\n")

    feature_cols = ["GRE Score", "TOEFL Score", "CGPA", "Research"]
    X = df[feature_cols].copy()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    print(f"Fitur       : {feature_cols}")
    print(f"Normalisasi : StandardScaler (Z-score)")
    print(f"Jumlah Data : {len(X)}")

    print("\nMetode Elbow:")
    inertias = []
    sil_scores = []
    K_range = range(2, 8)

    for k in K_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(X_scaled)
        inertias.append(km.inertia_)
        sil_scores.append(silhouette_score(X_scaled, km.labels_))

    print(f"  {'K':>3s}  |  {'Inertia':>10s}  |  {'Silhouette':>10s}")
    for k, iner, sil in zip(K_range, inertias, sil_scores):
        marker = " <- optimal" if sil == max(sil_scores) else ""
        print(f"  {k:3d}  |  {iner:10.2f}  |  {sil:10.4f}{marker}")

    best_k = list(K_range)[sil_scores.index(max(sil_scores))]
    print(f"\nK optimal berdasarkan Silhouette Score: {best_k}")

    print(f"\nClustering dengan K=2 (sesuai jurnal):")
    kmeans_model = KMeans(n_clusters=2, random_state=42, n_init=10)
    kmeans_model.fit(X_scaled)
    labels = kmeans_model.labels_

    df_result = df.copy()
    df_result["Cluster"] = labels

    for c in range(2):
        cluster_data = df_result[df_result["Cluster"] == c]
        mayoritas = cluster_data["Graduated"].mode()[0]
        print(f"  Cluster {c} -> Mayoritas: {mayoritas} ({len(cluster_data)} data)")

    print("\nHasil Clustering:")
    print(df_result.to_string(index=False))

    print("\nCentroid (sebelum denormalisasi):")
    centroid_scaled = kmeans_model.cluster_centers_
    centroid_df = pd.DataFrame(centroid_scaled, columns=feature_cols)
    centroid_df.index = [f"Cluster {i}" for i in range(2)]
    print(centroid_df.to_string())

    print("\nCentroid (setelah denormalisasi):")
    centroid_original = scaler.inverse_transform(centroid_scaled)
    centroid_orig_df = pd.DataFrame(centroid_original, columns=feature_cols)
    centroid_orig_df.index = [f"Cluster {i}" for i in range(2)]
    print(centroid_orig_df.round(2).to_string())

    print("\nDistribusi Label per Cluster:")
    for c in range(2):
        cluster_data = df_result[df_result["Cluster"] == c]
        print(f"\n  Cluster {c} ({len(cluster_data)} data):")
        for label, count in cluster_data["Graduated"].value_counts().items():
            persen = count / len(cluster_data) * 100
            print(f"    {label} : {count} ({persen:.1f}%)")

    sil = silhouette_score(X_scaled, labels)
    print(f"\nSilhouette Score (K=2): {sil:.4f}")

    return kmeans_model, scaler, feature_cols


# Menu 5 - Visualisasi K-Means
def visualisasi_kmeans(df):
    print("\n--- VISUALISASI K-MEANS ---\n")

    try:
        import matplotlib
        matplotlib.use("TkAgg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib belum terinstall. Install dengan: pip install matplotlib")
        return

    feature_cols = ["GRE Score", "TOEFL Score", "CGPA", "Research"]
    X = df[feature_cols].copy()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    kmeans_model = KMeans(n_clusters=2, random_state=42, n_init=10)
    kmeans_model.fit(X_scaled)
    labels = kmeans_model.labels_

    print("Membuka jendela grafik...")

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    colors = ["#FF6B6B", "#4ECDC4"]
    centroid_orig = scaler.inverse_transform(kmeans_model.cluster_centers_)

    pairs = [
        (0, 1, "GRE Score", "TOEFL Score"),
        (0, 2, "GRE Score", "CGPA"),
        (1, 2, "TOEFL Score", "CGPA"),
    ]
    for ax, (i, j, xlabel, ylabel) in zip(axes, pairs):
        for c in range(2):
            mask = labels == c
            ax.scatter(df[xlabel][mask], df[ylabel][mask],
                       c=colors[c], label=f"Cluster {c}", s=80,
                       edgecolors="white", linewidth=0.5, alpha=0.8)
        ax.scatter(centroid_orig[:, i], centroid_orig[:, j],
                   c="black", marker="X", s=200, label="Centroid",
                   edgecolors="gold", linewidth=2)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(f"{xlabel} vs {ylabel}")
        ax.legend()
        ax.grid(True, alpha=0.3)

    fig.suptitle("K-Means Clustering", fontsize=14, fontweight="bold")
    plt.tight_layout()

    save_path = os.path.join(SCRIPT_DIR, "kmeans_visual.png")
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Gambar disimpan: {save_path}")
    plt.show()

    # Elbow chart
    inertias = []
    K_range = range(2, 8)
    for k in K_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(X_scaled)
        inertias.append(km.inertia_)

    fig2, ax2 = plt.subplots(figsize=(8, 5))
    ax2.plot(list(K_range), inertias, "bo-", linewidth=2, markersize=8)
    ax2.set_xlabel("Jumlah Cluster (K)")
    ax2.set_ylabel("Inertia")
    ax2.set_title("Elbow Method")
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()

    save_path2 = os.path.join(SCRIPT_DIR, "elbow_chart.png")
    fig2.savefig(save_path2, dpi=150, bbox_inches="tight")
    print(f"Gambar disimpan: {save_path2}")
    plt.show()


# Menu 6 - Prediksi Data Baru
def prediksi_data_baru(df):
    print("\n--- PREDIKSI DATA BARU ---\n")

    le = LabelEncoder()
    df_encoded = df.copy()
    df_encoded["Graduated"] = le.fit_transform(df_encoded["Graduated"])

    feature_cols = ["GRE Score", "TOEFL Score", "CGPA", "Research"]
    X = df_encoded[feature_cols]
    y = df_encoded["Graduated"]

    dt_model = DecisionTreeClassifier(criterion="entropy", max_depth=5, random_state=42)
    dt_model.fit(X, y)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    kmeans_model = KMeans(n_clusters=2, random_state=42, n_init=10)
    kmeans_model.fit(X_scaled)

    df_temp = df.copy()
    df_temp["Cluster"] = kmeans_model.labels_
    cluster_labels = {}
    for c in range(2):
        cluster_labels[c] = df_temp[df_temp["Cluster"] == c]["Graduated"].mode()[0]

    print("Masukkan data calon mahasiswa:")
    try:
        gre = int(input("  GRE Score  (260-340) : "))
        toefl = int(input("  TOEFL Score (0-120)  : "))
        cgpa = float(input("  CGPA        (1-10)   : "))
        research = int(input("  Research    (0/1)    : "))
    except ValueError:
        print("Input tidak valid!")
        return

    data_baru = pd.DataFrame({
        "GRE Score": [gre], "TOEFL Score": [toefl],
        "CGPA": [cgpa], "Research": [research]
    })

    # Decision Tree
    dt_pred = dt_model.predict(data_baru)
    dt_label = le.inverse_transform(dt_pred)[0]
    dt_proba = dt_model.predict_proba(data_baru)[0]

    print(f"\nHasil Decision Tree: {dt_label}")
    for i, cls in enumerate(le.classes_):
        print(f"  Probabilitas {cls} : {dt_proba[i]*100:.1f}%")

    # K-Means
    data_baru_scaled = scaler.transform(data_baru)
    km_pred = kmeans_model.predict(data_baru_scaled)
    km_cluster = km_pred[0]
    km_label = cluster_labels[km_cluster]

    print(f"\nHasil K-Means: Cluster {km_cluster} -> {km_label}")
    distances = kmeans_model.transform(data_baru_scaled)[0]
    for i, d in enumerate(distances):
        print(f"  Jarak ke Cluster {i} : {d:.4f}")

    print(f"\nRangkuman:")
    print(f"  Decision Tree : {dt_label}")
    print(f"  K-Means       : {km_label}")
    if dt_label == km_label:
        print(f"  Kedua metode sepakat: {dt_label}")
    else:
        print("  Kedua metode berbeda. Disarankan ikuti Decision Tree (supervised).")


# Menu 7 - Perbandingan
def perbandingan_metode(df):
    print("\n--- PERBANDINGAN DECISION TREE vs K-MEANS ---\n")

    le = LabelEncoder()
    df_encoded = df.copy()
    df_encoded["Graduated"] = le.fit_transform(df_encoded["Graduated"])

    feature_cols = ["GRE Score", "TOEFL Score", "CGPA", "Research"]
    X = df_encoded[feature_cols]
    y = df_encoded["Graduated"]

    # Decision Tree
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    dt_model = DecisionTreeClassifier(criterion="entropy", max_depth=5, random_state=42)
    dt_model.fit(X_train, y_train)
    y_pred_dt = dt_model.predict(X_test)
    akurasi_dt = accuracy_score(y_test, y_pred_dt) * 100

    print("Decision Tree:")
    print(f"  Akurasi: {akurasi_dt:.2f}%")
    print(classification_report(y_test, y_pred_dt, target_names=le.classes_))

    # K-Means
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    kmeans_model = KMeans(n_clusters=2, random_state=42, n_init=10)
    kmeans_model.fit(X_scaled)

    df_temp = df.copy()
    df_temp["Cluster"] = kmeans_model.labels_
    cluster_mapping = {}
    for c in range(2):
        mayoritas = df_temp[df_temp["Cluster"] == c]["Graduated"].mode()[0]
        cluster_mapping[c] = le.transform([mayoritas])[0]

    y_pred_km = np.array([cluster_mapping[c] for c in kmeans_model.labels_])
    akurasi_km = accuracy_score(y, y_pred_km) * 100
    sil_score = silhouette_score(X_scaled, kmeans_model.labels_)

    print("K-Means:")
    print(f"  Akurasi (mapping): {akurasi_km:.2f}%")
    print(f"  Silhouette Score : {sil_score:.4f}")
    print(classification_report(y, y_pred_km, target_names=le.classes_))

    print("Perbandingan:")
    print(f"  {'Metrik':<25s} | {'Decision Tree':>15s} | {'K-Means':>15s}")
    print(f"  {'Tipe':<25s} | {'Supervised':>15s} | {'Unsupervised':>15s}")
    print(f"  {'Akurasi':<25s} | {akurasi_dt:>14.2f}% | {akurasi_km:>14.2f}%")
    print(f"  {'Butuh Label?':<25s} | {'Ya':>15s} | {'Tidak':>15s}")

    if akurasi_dt >= akurasi_km:
        print(f"\nDecision Tree unggul dengan akurasi {akurasi_dt:.2f}%")
    else:
        print(f"\nK-Means unggul dengan akurasi {akurasi_km:.2f}%")


def menu_utama():
    df = load_dataset()

    while True:
        print("\n===================================")
        print("PREDIKSI PENERIMAAN MAHASISWA")
        print("PASCASARJANA - Universitas Indonesia")
        print("Metode: K-Means & Decision Tree")
        print("===================================")
        print("[1] Tampilkan Dataset")
        print("[2] Analisis Decision Tree")
        print("[3] Visualisasi Decision Tree")
        print("[4] Analisis K-Means Clustering")
        print("[5] Visualisasi K-Means")
        print("[6] Prediksi Data Baru")
        print("[7] Perbandingan DT vs K-Means")
        print("[0] Keluar")

        pilihan = input("\nPilih menu: ").strip()

        if pilihan == "1":
            tampilkan_dataset(df)
        elif pilihan == "2":
            decision_tree_analysis(df)
        elif pilihan == "3":
            visualisasi_decision_tree(df)
        elif pilihan == "4":
            kmeans_analysis(df)
        elif pilihan == "5":
            visualisasi_kmeans(df)
        elif pilihan == "6":
            prediksi_data_baru(df)
        elif pilihan == "7":
            perbandingan_metode(df)
        elif pilihan == "0":
            print("Program selesai.")
            break
        else:
            print("Pilihan tidak valid.")

        if pilihan in ["1","2","3","4","5","6","7"]:
            input("\nTekan Enter untuk kembali ke menu...")


if __name__ == "__main__":
    menu_utama()
