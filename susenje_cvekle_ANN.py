# =============================================================================
# МОДЕЛОВАЊЕ КИНЕТИКЕ СУШЕЊА ЦВЕKЛЕ ПОМОЋУ НЕУРОНСКИХ МРЕЖА
# Програмски језик: Python 3
# Библиотеке: TensorFlow/Keras, scikit-learn, pandas, matplotlib
# =============================================================================

import pandas as pd
import numpy as np  
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = 'DejaVu Sans'

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("  МОДЕЛОВАЊЕ КИНЕТИКЕ СУШЕЊА ЦВЕКЛЕ - ANN")
print("=" * 60)
print(f"  TensorFlow верзија: {tf.__version__}")
print()

# =============================================================================
# 1. УЧИТАВАЊЕ И ПРИПРЕМА ПОДАТАКА
# =============================================================================
print("--- 1. УЧИТАВАЊЕ ПОДАТАКА ---")

xl = pd.read_excel('cipsCvekla.xlsx', sheet_name='Sheet1', header=None)

# Дефиниција 8 тестова: (почетна колона, назив, бланширан, дебљина)
testovi = [
    (1,  "Свежа, танко (70°C)",                 0, 0),
    (6,  "Свежа, танко (80°C+55°C)",            0, 0),
    (11, "Свежа, танко (80°C+65°C+55°C)",       0, 0),
    (16, "Свежа, дебље (80°C+65°C+55°C)",      0, 1),
    (21, "Бланш., танко (70°C)",                 1, 0),
    (26, "Бланш., танко (80°C+55°C)",            1, 0),
    (31, "Бланш., танко (80°C+65°C+55°C)",       1, 0),
    (36, "Бланш., дебље (80°C+65°C+55°C)",      1, 1),
]

all_data = []

for start_col, naziv, blansiran, debljina in testovi:
    rows = xl.iloc[5:, [start_col, start_col+2, start_col+3]].copy()
    rows.columns = ['vreme', 'masa', 'temp']

    # Задржи само нумеричке редове
    rows = rows[pd.to_numeric(rows['vreme'], errors='coerce').notna()]
    rows = rows[pd.to_numeric(rows['masa'],  errors='coerce').notna()]
    rows['vreme'] = pd.to_numeric(rows['vreme'])
    rows['masa']  = pd.to_numeric(rows['masa'])
    rows['temp']  = pd.to_numeric(rows['temp'], errors='coerce')

    # Попуни недостајуће вредности температуре (t=0 нема мерење)
    rows['temp'] = rows['temp'].ffill().bfill()

    masa0     = rows.loc[rows['vreme'] == 0, 'masa'].values[0]
    masa_suva = rows['masa'].min()

    if masa0 == masa_suva:
        continue

    # Рачунање односа влаге (Moisture Ratio – MR)
    rows['MR']        = (rows['masa'] - masa_suva) / (masa0 - masa_suva)
    rows['blansiran'] = blansiran
    rows['debljina']  = debljina
    rows['test_id']   = naziv

    all_data.append(rows[['vreme','temp','MR','blansiran','debljina','test_id']])

df = pd.concat(all_data, ignore_index=True)
df = df[df['MR'] >= 0].copy()

print(f"  Укупно мерења: {len(df)}")
print(f"  Број тестова: {df['test_id'].nunique()}")
print()
print("  Расподела по тестовима:")
for naziv, n in df.groupby('test_id').size().items():
    print(f"    {naziv}: {n} мерења")
print()

# =============================================================================
# 2. ДЕФИНИЦИЈА УЛАЗНИХ И ИЗЛАЗНИХ ВАРИЈАБЛИ
# =============================================================================
print("--- 2. ДЕФИНИЦИЈА ВАРИЈАБЛИ ---")

# Улазне варијабле (X): Вријеме, Температура, Бланширан, Дебљина
# Излазна варијабла (y): Однос влаге MR
feature_cols = ['vreme', 'temp', 'blansiran', 'debljina']
target_col   = 'MR'

X = df[feature_cols].values
y = df[target_col].values

print(f"  Улазне варијабле: {feature_cols}")
print(f"  Излазна варијабла: {target_col}")
print(f"  Облик X: {X.shape}, облик y: {y.shape}")
print()

# =============================================================================
# 3. ПОДЕЛА ПОДАТАКА И НОРМАЛИЗАЦИЈА
# =============================================================================
print("--- 3. ПОДЕЛА И НОРМАЛИЗАЦИЈА ---")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler_X = StandardScaler()
X_train_sc = scaler_X.fit_transform(X_train)
X_test_sc  = scaler_X.transform(X_test)

print(f"  Скуп за тренирање: {len(X_train)} узорака ({100*(1-0.2):.0f}%)")
print(f"  Скуп за тестирање: {len(X_test)} узорака ({100*0.2:.0f}%)")
print()

# =============================================================================
# 4. ГРАДЊА MLP МОДЕЛА (Вишеслојни перцептрон)
# =============================================================================
print("--- 4. АРХИТЕКТУРА MLP МОДЕЛА ---")

def build_mlp(hidden_layers, neurons, dropout_rate=0.2, lr=0.001):
    """
    Гради MLP модел са задатим бројем скривених слојева и неурона.
    
    Параметри:
        hidden_layers : int   – број скривених слојева
        neurons       : int   – број неурона по слоју
        dropout_rate  : float – стопа dropout регуларизације
        lr            : float – стопа учења (learning rate)
    """
    model = keras.Sequential(name="MLP_kinetika_susenja")

    # Улазни слој
    model.add(layers.Input(shape=(X_train_sc.shape[1],)))

    # Скривени слојеви
    for i in range(hidden_layers):
        model.add(layers.Dense(neurons, activation='relu',
                               kernel_initializer='he_normal',
                               name=f"Dense_{i+1}"))
        model.add(layers.BatchNormalization())
        model.add(layers.Dropout(dropout_rate))

    # Излазни слој – регресија (без активацијске функције)
    model.add(layers.Dense(1, activation='linear', name="Izlaz"))

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss='mse',
        metrics=['mae']
    )
    return model

# Изграђујемо главни модел: 3 скривена слоја по 64 неурона
model = build_mlp(hidden_layers=3, neurons=64, dropout_rate=0.2, lr=0.001)
model.summary()
print()

# =============================================================================
# 5. ТРЕНИРАЊЕ МОДЕЛА
# =============================================================================
print("--- 5. ТРЕНИРАЊЕ МОДЕЛА ---")

callbacks = [
    EarlyStopping(
        monitor='val_loss',
        patience=30,
        restore_best_weights=True,
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=15,
        min_lr=1e-6,
        verbose=1
    )
]

history = model.fit(
    X_train_sc, y_train,
    validation_split=0.15,
    epochs=500,
    batch_size=16,
    callbacks=callbacks,
    verbose=1
)

print(f"\n  Тренирање завршено. Укупно епоха: {len(history.history['loss'])}")
print()

# =============================================================================
# 6. ЕВАЛУАЦИЈА МОДЕЛА
# =============================================================================
print("--- 6. ЕВАЛУАЦИЈА МОДЕЛА ---")

y_pred_train = model.predict(X_train_sc, verbose=0).flatten()
y_pred_test  = model.predict(X_test_sc,  verbose=0).flatten()

def metrics(y_true, y_pred, naziv):
    mse  = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae  = mean_absolute_error(y_true, y_pred)
    r2   = r2_score(y_true, y_pred)
    print(f"  [{naziv}]")
    print(f"    R²   = {r2:.4f}")
    print(f"    RMSE = {rmse:.4f}")
    print(f"    MAE  = {mae:.4f}")
    print(f"    MSE  = {mse:.6f}")
    return r2, rmse, mae

r2_tr, rmse_tr, mae_tr = metrics(y_train, y_pred_train, "Скуп за тренирање")
print()
r2_ts, rmse_ts, mae_ts = metrics(y_test,  y_pred_test,  "Скуп за тестирање")
print()

# =============================================================================
# 7. ВИЗУАЛИЗАЦИЈА – 4 графика
# =============================================================================
print("--- 7. ВИЗУАЛИЗАЦИЈА ---")

fig, axes = plt.subplots(2, 2, figsize=(14, 11))
fig.suptitle("Моделовање кинетике сушења цвекле – ANN (MLP)", fontsize=14, fontweight='bold')

# --- График 1: Губитак (Loss) током тренирања ---
ax = axes[0, 0]
ax.plot(history.history['loss'],     label='Тренирање', color='steelblue', linewidth=2)
ax.plot(history.history['val_loss'], label='Валидација', color='tomato',    linewidth=2, linestyle='--')
ax.set_title('Промена губитка (MSE) током тренирања')
ax.set_xlabel('Епоха')
ax.set_ylabel('Губитак (MSE)')
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_yscale('log')

# --- График 2: Предвиђене vs. стварне вредности ---
ax = axes[0, 1]
all_true = np.concatenate([y_train, y_test])
all_pred = np.concatenate([y_pred_train, y_pred_test])
ax.scatter(y_train, y_pred_train, alpha=0.5, s=20, color='steelblue', label=f'Тренирање (R²={r2_tr:.3f})')
ax.scatter(y_test,  y_pred_test,  alpha=0.7, s=25, color='tomato',    label=f'Тестирање (R²={r2_ts:.3f})', marker='^')
ax.plot([0, 1], [0, 1], 'k--', linewidth=1.5, label='Идеалан модел')
ax.set_xlabel('Измерени MR')
ax.set_ylabel('Предвиђени MR')
ax.set_title('Предвиђене vs. измерене вредности MR')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
ax.set_xlim(-0.05, 1.1)
ax.set_ylim(-0.05, 1.1)

# --- График 3: Криве сушења по тестовима ---
ax = axes[1, 0]
boje = plt.cm.tab10(np.linspace(0, 1, 8))

for idx, (naziv, grupa) in enumerate(df.groupby('test_id')):
    grupa_s = grupa.sort_values('vreme')
    X_g = scaler_X.transform(grupa_s[feature_cols].values)
    MR_pred = model.predict(X_g, verbose=0).flatten()

    ax.plot(grupa_s['vreme'].values, grupa_s['MR'].values,
            'o', color=boje[idx], markersize=4, alpha=0.6)
    ax.plot(grupa_s['vreme'].values, MR_pred,
            '-', color=boje[idx], linewidth=1.8, label=naziv)

ax.set_xlabel('Вријеме (мин)')
ax.set_ylabel('Однос влаге – MR (-)')
ax.set_title('Криве сушења: измерено (тачке) vs. предвиђено (линија)')
ax.legend(fontsize=7, loc='upper right')
ax.grid(True, alpha=0.3)
ax.set_ylim(-0.05, 1.1)

# --- График 4: Резидуали ---
ax = axes[1, 1]
residuali_train = y_train - y_pred_train
residuali_test  = y_test  - y_pred_test
ax.scatter(y_pred_train, residuali_train, alpha=0.4, s=20, color='steelblue', label='Тренирање')
ax.scatter(y_pred_test,  residuali_test,  alpha=0.6, s=25, color='tomato',    label='Тестирање', marker='^')
ax.axhline(y=0, color='black', linewidth=1.5, linestyle='--')
ax.set_xlabel('Предвиђени MR')
ax.set_ylabel('Резидуал (Измерено – Предвиђено)')
ax.set_title('Дијаграм резидуала')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('rezultati_ANN.png', dpi=150, bbox_inches='tight')
plt.show()
print("  График сачуван: rezultati_ANN.png")
print()

# =============================================================================
# 8. ПОРЕЂЕЊЕ АРХИТЕКТУРА (Хиперпараметарска анализа)
# =============================================================================
print("--- 8. ПОРЕЂЕЊЕ РАЗЛИЧИТИХ АРХИТЕКТУРА ---")

arhitekture = [
    {"slojevi": 1, "neuroni": 32,  "naziv": "1×32"},
    {"slojevi": 2, "neuroni": 32,  "naziv": "2×32"},
    {"slojevi": 2, "neuroni": 64,  "naziv": "2×64"},
    {"slojevi": 3, "neuroni": 64,  "naziv": "3×64"},  # ← главни модел
    {"slojevi": 3, "neuroni": 128, "naziv": "3×128"},
    {"slojevi": 4, "neuroni": 64,  "naziv": "4×64"},
]

rezultati_arhitektura = []

for arh in arhitekture:
    m = build_mlp(arh["slojevi"], arh["neuroni"], dropout_rate=0.2, lr=0.001)
    cb = [EarlyStopping(monitor='val_loss', patience=25, restore_best_weights=True, verbose=0)]
    m.fit(X_train_sc, y_train,
          validation_split=0.15,
          epochs=300,
          batch_size=16,
          callbacks=cb,
          verbose=0)
    yp = m.predict(X_test_sc, verbose=0).flatten()
    r2   = r2_score(y_test, yp)
    rmse = np.sqrt(mean_squared_error(y_test, yp))
    print(f"  Архитектура {arh['naziv']:8s} → R² = {r2:.4f},  RMSE = {rmse:.4f}")
    rezultati_arhitektura.append({"Arhitektura": arh["naziv"], "R²": r2, "RMSE": rmse})

df_arh = pd.DataFrame(rezultati_arhitektura)
print()
print("  Резиме архитектура:")
print(df_arh.to_string(index=False))
print()

# Бар график поређења
fig2, axes2 = plt.subplots(1, 2, figsize=(12, 5))
fig2.suptitle("Поређење архитектура MLP модела", fontsize=13, fontweight='bold')

axes2[0].bar(df_arh["Arhitektura"], df_arh["R²"], color='steelblue', edgecolor='black')
axes2[0].set_title("Коефицијент детерминације R²")
axes2[0].set_ylabel("R²")
axes2[0].set_ylim(0, 1.05)
axes2[0].axhline(y=0.99, color='red', linestyle='--', alpha=0.5, label='R²=0.99')
axes2[0].legend()
axes2[0].grid(True, alpha=0.3, axis='y')
for i, v in enumerate(df_arh["R²"]):
    axes2[0].text(i, v + 0.01, f"{v:.3f}", ha='center', fontsize=9)

axes2[1].bar(df_arh["Arhitektura"], df_arh["RMSE"], color='tomato', edgecolor='black')
axes2[1].set_title("Корен средње квадратне грешке (RMSE)")
axes2[1].set_ylabel("RMSE")
axes2[1].grid(True, alpha=0.3, axis='y')
for i, v in enumerate(df_arh["RMSE"]):
    axes2[1].text(i, v + 0.001, f"{v:.4f}", ha='center', fontsize=9)

plt.tight_layout()
plt.savefig('poređenje_arhitektura.png', dpi=150, bbox_inches='tight')
plt.show()
print("  График сачуван: poređenje_arhitektura.png")
print()

# =============================================================================
# 9. ФИНАЛНИ РЕЗИМЕ
# =============================================================================
print("=" * 60)
print("  РЕЗИМЕ РЕЗУЛТАТА – ГЛАВНИ МОДЕЛ (3×64)")
print("=" * 60)
print(f"  Улазне варијабле : Вријеме, Температура, Бланширан, Дебљина")
print(f"  Архитектура      : 4 → 64 → 64 → 64 → 1")
print(f"  Активациона f.   : ReLU (скривени), Linear (излаз)")
print(f"  Оптимизатор      : Adam (lr=0.001)")
print(f"  Регуларизација   : Dropout (0.2) + BatchNorm")
print()
print(f"  Скуп за тренирање ({len(X_train)} узорака):")
print(f"    R²   = {r2_tr:.4f}")
print(f"    RMSE = {rmse_tr:.4f}")
print(f"    MAE  = {mae_tr:.4f}")
print()
print(f"  Скуп за тестирање ({len(X_test)} узорака):")
print(f"    R²   = {r2_ts:.4f}")
print(f"    RMSE = {rmse_ts:.4f}")
print(f"    MAE  = {mae_ts:.4f}")
print()
print("  Сачуване датотеке:")
print("    → rezultati_ANN.png")
print("    → poređenje_arhitektura.png")
print("=" * 60)
