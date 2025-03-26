####################################################
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import RobustScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import xgboost as xgb
import seaborn as sns
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

pd.set_option('display.max_row', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width',500)

# Veri setini yükleme
df = pd.read_csv("datasets/veri_seti.csv")
df.head()
# Tarih sütununu datetime formatına çevirme
df["Olus_tarihi"] = pd.to_datetime(df["Olus_tarihi"], errors='coerce')

# Marmara Bölgesi verilerini filtreleme
marmara_enlem_min, marmara_enlem_max = 37.90, 41.10
marmara_boylam_min, marmara_boylam_max = 26, 31

df_marmara = df[
    (df['Enlem'] >= marmara_enlem_min) & (df['Enlem'] <= marmara_enlem_max) &
    (df['Boylam'] >= marmara_boylam_min) & (df['Boylam'] <= marmara_boylam_max)
]


df = df_marmara
df = df[df['Olus_tarihi'] >= "1980-1-1"]

# Tarih bileşenlerini ayırma
df['Year'] = df['Olus_tarihi'].dt.year
df['Month'] = df['Olus_tarihi'].dt.month
df['Day'] = df['Olus_tarihi'].dt.day
df['Julian_Date'] = df['Olus_tarihi'].apply(lambda x: x.to_julian_date())

df.describe().T
##################
df_filtered = df[df["xM"] >= 3]
df_filtered["Yil"] = df_filtered["Olus_tarihi"].dt.year
df_filtered["Yil_Grubu"] = (df_filtered["Yil"] // 5) * 5  # 5 yıllık aralıklar

# 5 yıllık aralıklarda toplam deprem sayısını hesaplama
yil_aralik_siklik = df_filtered["Yil_Grubu"].value_counts().sort_index()

# Sonuçları yazdırma
print(yil_aralik_siklik)

# Görselleştirme (Bar Grafiği)
plt.figure(figsize=(12, 6))
yil_aralik_siklik.plot(kind="bar", color="skyblue", edgecolor="black")
plt.title("M ≥ 3 Deprem Sıklığı", fontsize=16)
plt.xlabel("Yıl Aralığı", fontsize=12)
plt.ylabel("Deprem Sayısı", fontsize=12)
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.xticks(rotation=45)
plt.show(block = True)
##################
# Kategorik verileri LabelEncoder ile dönüştürme
label_encoder = LabelEncoder()
for column in df.select_dtypes(include=['object']).columns:
    df[column] = label_encoder.fit_transform(df[column])

# Hedef değişken ve özellikleri ayırma
X = df.drop(["xM", "Olus_tarihi"], axis=1)
y = df["xM"]

# Eğitim ve test setlerine ayırma
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train.shape
X_test.shape
y_train.shape
y_test.shape

# Veriyi ölçeklendirme
scaler = RobustScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Decision Tree Modeli
dt_model = DecisionTreeRegressor()
dt_model.fit(X_train, y_train)
dt_predictions = dt_model.predict(X_test)
dt_mse = mean_squared_error(y_test, dt_predictions)
dt_rmse = np.sqrt(dt_mse)
dt_mae = mean_absolute_error(y_test, dt_predictions)

# RandomForest Modeli
rf_model = RandomForestRegressor()
rf_model.fit(X_train, y_train)
rf_predictions = rf_model.predict(X_test)
rf_mse = mean_squared_error(y_test, rf_predictions)
rf_rmse = np.sqrt(rf_mse)
rf_mae = mean_absolute_error(y_test, rf_predictions)

# XGBoost Modeli
xgb_model = xgb.XGBRegressor(objective='reg:squarederror')
xgb_model.fit(X_train, y_train)
xgb_predictions = xgb_model.predict(X_test)
xgb_mse = mean_squared_error(y_test, xgb_predictions)
xgb_rmse = np.sqrt(xgb_mse)
xgb_mae = mean_absolute_error(y_test, xgb_predictions)

print(f"Decision Tree - MSE: {dt_mse:.4f}, RMSE: {dt_rmse:.4f}, MAE: {dt_mae:.4f}")
print(f"RandomForest - MSE: {rf_mse:.4f}, RMSE: {rf_rmse:.4f}, MAE: {rf_mae:.4f}")
print(f"XGBoost - MSE: {xgb_mse:.4f}, RMSE: {xgb_rmse:.4f}, MAE: {xgb_mae:.4f}")

df.describe().T


# Plotting the results for each model

# Decision Tree için grafik
plt.figure(figsize=(10, 6))
plt.bar(['MSE', 'RMSE', 'MAE'], [dt_mse, dt_rmse, dt_mae], color=['blue', 'orange', 'green'])
plt.xlabel('Metrics')
plt.ylabel('Values')
plt.title('Decision Tree - MSE, RMSE and MAE')
plt.ylim(0, max(dt_mse, dt_rmse, dt_mae) * 1.5)  # Y ekseni aralığını artırma
plt.grid(True)
plt.show(block = True)

# RandomForest için grafik
plt.figure(figsize=(10, 6))
plt.bar(['MSE', 'RMSE', 'MAE'], [rf_mse, rf_rmse, rf_mae], color=['blue', 'orange', 'green'])
plt.xlabel('Metrics')
plt.ylabel('Values')
plt.title('RandomForest - MSE, RMSE and MAE')
plt.ylim(0, max(rf_mse, rf_rmse, rf_mae) * 1.5)  # Y ekseni aralığını artırma
plt.grid(True)
plt.show(block = True)

# XGBoost için grafik
plt.figure(figsize=(10, 6))
plt.bar(['MSE', 'RMSE', 'MAE'], [xgb_mse, xgb_rmse, xgb_mae], color=['blue', 'orange', 'green'])
plt.xlabel('Metrics')
plt.ylabel('Values')
plt.title('XGBoost - MSE, RMSE and MAE')
plt.ylim(0, max(xgb_mse, xgb_rmse, xgb_mae) * 1.5)  # Y ekseni aralığını artırma
plt.grid(True)
plt.show(block = True)

# Parametre grid'i
dt_param_grid = {
    'max_depth': [5, 10, 15, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Grid SearchCV
dt_grid_search = GridSearchCV(
    DecisionTreeRegressor(),
    param_grid=dt_param_grid,
    cv=5,
    scoring='neg_mean_squared_error',
    n_jobs=-1
)

# Eğitim ve test verileri
dt_grid_search.fit(X_train, y_train)

# En iyi modeli ve tahminleri al
best_dt_model = dt_grid_search.best_estimator_
best_dt_predictions = best_dt_model.predict(X_test)

# Hataları hesapla
best_dt_mse = mean_squared_error(y_test, best_dt_predictions)
best_dt_rmse = np.sqrt(best_dt_mse)
best_dt_mae = mean_absolute_error(y_test, best_dt_predictions)



# Random Forest için Hiperparametre Optimizasyonu
rf_param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [5, 10, 20, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}
rf_random_search = RandomizedSearchCV(
    RandomForestRegressor(),
    param_distributions=rf_param_grid,
    n_iter=50,
    cv=5,
    scoring='neg_mean_squared_error',
    n_jobs=-1,
    random_state=42
)
rf_random_search.fit(X_train, y_train)

best_rf_model = rf_random_search.best_estimator_
best_rf_predictions = best_rf_model.predict(X_test)
best_rf_mse = mean_squared_error(y_test, best_rf_predictions)
best_rf_rmse = np.sqrt(best_rf_mse)
best_rf_mae = mean_absolute_error(y_test, best_rf_predictions)



# XGBoost için Hiperparametre Optimizasyonu
xgb_param_grid = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 5, 10],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0]
}
xgb_random_search = RandomizedSearchCV(
    xgb.XGBRegressor(objective='reg:squarederror'),
    param_distributions=xgb_param_grid,
    n_iter=50,
    cv=5,
    scoring='neg_mean_squared_error',
    n_jobs=-1,
    random_state=42
)
xgb_random_search.fit(X_train, y_train)

best_xgb_model = xgb_random_search.best_estimator_
best_xgb_predictions = best_xgb_model.predict(X_test)
best_xgb_mse = mean_squared_error(y_test, best_xgb_predictions)
best_xgb_rmse = np.sqrt(best_xgb_mse)
best_xgb_mae = mean_absolute_error(y_test, best_xgb_predictions)

print(f"Best Decision Tree - MSE: {best_dt_mse:.4f}, RMSE: {best_dt_rmse:.4f}, MAE: {best_dt_mae:.4f}")
print(f"Best Random Forest - MSE: {best_rf_mse:.4f}, RMSE: {best_rf_rmse:.4f}, MAE: {best_rf_mae:.4f}")
print(f"Best XGBoost - MSE: {best_xgb_mse:.4f}, RMSE: {best_xgb_rmse:.4f}, MAE: {best_xgb_mae:.4f}")




models = ['Decision Tree', 'Random Forest', 'XGBoost']
metrics = ['MSE', 'RMSE', 'MAE']

# Önceki sonuçlar (hiperparametre optimizasyonu yapılmamış)
before_mse = [dt_mse, rf_mse, xgb_mse]
before_rmse = [dt_rmse, rf_rmse, xgb_rmse]
before_mae = [dt_mae, rf_mae, xgb_mae]

# Sonraki sonuçlar (hiperparametre optimizasyonu yapılmış)
after_mse = [best_dt_mse, best_rf_mse, best_xgb_mse]
after_rmse = [best_dt_rmse, best_rf_rmse, best_xgb_rmse]
after_mae = [best_dt_mae, best_rf_mae, best_xgb_mae]

# Plot için bir fonksiyon tanımlama
def plot_comparison(metric_values_before, metric_values_after, metric_name):
    x = np.arange(len(models))  # Model sayısı kadar x ekseni indeksi
    width = 0.35  # Sütun genişliği

    fig, ax = plt.subplots(figsize=(10, 6))
    rects1 = ax.bar(x - width/2, metric_values_before, width, label='Before Optimization', color='blue')
    rects2 = ax.bar(x + width/2, metric_values_after, width, label='After Optimization', color='orange')

    # Grafik ayarları
    ax.set_ylabel(metric_name)
    ax.set_xlabel('Models')
    ax.set_title(f'{metric_name} Before and After Hyperparameter Optimization')
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.legend()

    # Değer etiketlerini ekleme
    def add_labels(rects):
        for rect in rects:
            height = rect.get_height()
            ax.text(
                rect.get_x() + rect.get_width() / 2.,
                height,
                f'{height:.4f}',
                ha='center',
                va='bottom'
            )

    add_labels(rects1)
    add_labels(rects2)

    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    plt.show(block = True)

# Metriklerin karşılaştırmalı grafikleri
plot_comparison(before_mse, after_mse, 'MSE')
plot_comparison(before_rmse, after_rmse, 'RMSE')
plot_comparison(before_mae, after_mae, 'MAE')

######################################

df.head()

count_4_plus = df[df['xM'] >= 4].shape[0]
count_5_plus = df[df['xM'] >= 5].shape[0]
count_6_plus = df[df['xM'] >= 6].shape[0]
count_7_plus = df[df['xM'] >= 7].shape[0]

# Sonuçları yazdır
print(f"4 üzeri deprem sayısı: {count_4_plus}")
print(f"5 üzeri deprem sayısı: {count_5_plus}")
print(f"6 üzeri deprem sayısı: {count_6_plus}")
print(f"7 üzeri deprem sayısı: {count_7_plus}")

df_6_plus = df[df['xM'] >= 6]

# Seçilen depremleri göstermek
print(df_6_plus[['Olus_tarihi', 'Enlem', 'Boylam', 'xM', 'Der(km)']])