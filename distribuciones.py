import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

st.set_page_config(page_title="Distribuciones de Probabilidad", layout="centered")

st.title("Distribuciones de Probabilidad")

# Distribuciones y sus parámetros
distribuciones = {
    "Normal (μ, σ)": ("norm", ["μ", "σ"]),
    "Binomial (n, p)": ("binom", ["n", "p"]),
    "Poisson (λ)": ("poisson", ["λ"]),
    "Exponencial (λ)": ("expon", ["λ"]),
    "Uniforme continua (a, b)": ("uniform", ["a", "b"]),
    "Uniforme discreta (a, b)": ("randint", ["a", "b"]),
    "t de Student (df)": ("t", ["df"]),
    "F de Fisher (df1, df2)": ("f", ["df1", "df2"]),
    "Chi-cuadrada (df)": ("chi2", ["df"]),
}

# Selección de la distribución
dist_nombre = st.selectbox("Selecciona una distribución:", list(distribuciones.keys()))
dist_code, param_labels = distribuciones[dist_nombre]

# Ingreso de parámetros
params = []
for label in param_labels:
    valor = st.number_input(f"Ingresar {label}:", value=0.0000, step=0.0001)
    if label in ["n", "df", "df1", "df2"]:
        valor = int(valor)
    params.append(valor)

# Selección de valor x o probabilidad p
modo = st.radio("¿Qué deseas calcular?", ["Probabilidad P(X ≤ x)", "Inversa (x tal que P(X ≤ x) = p)"])

if modo == "Probabilidad P(X ≤ x)":
    x = st.number_input("Ingresa el valor de x:", value=0.0000, step=0.0001)
    p = None
else:
    p = st.number_input("Ingresa la probabilidad p (entre 0 y 1):", min_value=0.0000, max_value=1.0000, value=0.9500, step=0.0001)
    x = None

# Selección de tipo de distribución (continua, discreta)
discretas = ["binom", "poisson", "randint"]
discreta = dist_code in discretas

# Botón para ejecutar
if st.button("Calcular y Graficar"):
    if dist_code == "expon":
        distribucion = stats.expon(scale=1 / params[0])
    elif dist_code == "uniform":
        a, b = params
        distribucion = stats.uniform(loc=a, scale=(b - a))
    else:
        distribucion = getattr(stats, dist_code)(*params)

    if x is not None:
        prob = distribucion.cdf(x)
        fx = distribucion.pmf(x) if discreta else distribucion.pdf(x)
        st.write(f"**P(X ≤ {x}) = {prob:.4f}**")
        st.write(f"**f({x}) = {fx:.4f}**")
    elif p is not None:
        x = distribucion.ppf(p)
        st.write(f"**x tal que P(X ≤ x) = {p:.4f} es {x:.4f}**")

    x_min, x_max = distribucion.ppf(0.001), distribucion.ppf(0.999)
    x_vals = np.arange(np.floor(x_min), np.ceil(x_max) + 1) if discreta else np.linspace(x_min, x_max, 500)
    y_vals = distribucion.pmf(x_vals) if discreta else distribucion.pdf(x_vals)

    fig, ax = plt.subplots(figsize=(8, 4))
    if discreta:
        ax.vlines(x_vals, 0, y_vals, colors='blue')
        ax.scatter(x_vals, y_vals, color='blue')
        sombra = x_vals <= x
        ax.bar(x_vals[sombra], distribucion.pmf(x_vals[sombra]), color='skyblue', edgecolor='black')
    else:
        ax.plot(x_vals, y_vals, label=dist_code)
        ax.fill_between(x_vals, 0, y_vals, where=(x_vals <= x), color='skyblue', alpha=0.7)

    ax.axvline(x, color='red', linestyle='--', label=f'x = {x:.2f}')
    ax.set_title(f"Distribución {dist_nombre}")
    ax.set_xlabel("x")
    ax.set_ylabel("f(x)")
    ax.grid(True)
    ax.legend()
    st.pyplot(fig)
