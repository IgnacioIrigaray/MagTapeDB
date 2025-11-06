import pandas as pd
import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import plotly.express as px
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression
import numpy as np

# Cargar datos
df = pd.read_csv("Datos_ENF_DIapasonLA_CDM.csv")
df["fecha"] = pd.to_datetime(df["fecha"], errors="coerce")
df["mes"] = df["fecha"].dt.to_period("M").astype(str)
df = df.dropna(subset=["fecha", "frecuencia_hz", "enf", "presence_index"])
meses_unicos = sorted(df["mes"].dropna().unique().tolist())

# Velocidades normalizadas
df["vel_pyin"] = df["frecuencia_hz"] / 440
df["vel_enf"] = df["enf"] / 50

app = dash.Dash(__name__)
app.title = "Correlación ENF vs pyin"

presence_min = df["presence_index"].min()
presence_max = df["presence_index"].max()

app.layout = html.Div([
    html.H2("Correlación entre ENF y PYIN filtrada por Presence Index y Mes"),

    html.Label("Filtrar por Presence Index:"),
    dcc.RangeSlider(
        id='range-slider',
        min=presence_min,
        max=presence_max,
        step=0.1,
        value=[presence_min, presence_max],
        tooltip={"placement": "bottom", "always_visible": True}
    ),

    html.Label("Filtrar por Mes:"),
    dcc.Dropdown(
        id='mes-dropdown',
        options=[{"label": mes, "value": mes} for mes in meses_unicos],
        value=meses_unicos,
        multi=True
    ),

    html.Div([
        html.Label("Ancho del gráfico (px):"),
        dcc.Input(id='width-input', type='number', value=2100),
        html.Label("Alto del gráfico (px):"),
        dcc.Input(id='height-input', type='number', value=1500),
    ], style={"marginTop": "10px", "marginBottom": "10px"}),

    html.Div(id='correlation-output', style={'margin': '10px 0', 'fontSize': 18}),
    dcc.Graph(id='scatter-plot', config={"toImageButtonOptions": {"format": "svg"}}),
])

@app.callback(
    Output('scatter-plot', 'figure'),
    Output('correlation-output', 'children'),
    Input('range-slider', 'value'),
    Input('mes-dropdown', 'value'),
    Input('width-input', 'value'),
    Input('height-input', 'value')
)
def update_graph(range_vals, selected_meses, width, height):
    min_threshold, max_threshold = range_vals

    mask = (df["mes"].isin(selected_meses))
    all_filtered = df[mask]

    selected = all_filtered[
        (all_filtered["presence_index"] >= min_threshold) &
        (all_filtered["presence_index"] <= max_threshold)
    ]

    # Crear figura base con puntos no seleccionados (en gris)
    fig = px.scatter(
        all_filtered,
        x="vel_pyin", y="vel_enf",
        hover_data=["fecha"],
        opacity=0,
        color_discrete_sequence=["lightgray"],
        labels={"vel_pyin": "PYIN / 440", "vel_enf": "ENF / 50"}
    )

    if len(selected) > 1:
        x = selected["vel_pyin"].values.reshape(-1, 1)
        y = selected["vel_enf"].values
        corr, _ = pearsonr(x.flatten(), y)
        model = LinearRegression().fit(x, y)
        A = model.coef_[0]
        B = model.intercept_
        if A != 0:
            freq_pyin_nominal = 440 * (1 - B) / A
            est_text = f"f_pyin_nominal ≈ {freq_pyin_nominal:.2f} Hz"
        else:
            est_text = "Ajuste degenerado (pendiente cero)"

        corr_text = f"Correlación de Pearson: {corr:.3f} | A: {A:.3f} | B: {B:.3f} | {est_text}"

        # Agregar puntos seleccionados en color
        fig_sel = px.scatter(
            selected,
            x="vel_pyin", y="vel_enf",
            color="fecha",
            hover_data=["fecha"],
            #color_continuous_scale="viridis"
        ).update_traces(marker=dict(size=20))
        
        for trace in fig_sel.data:
            fig.add_trace(trace)

        # Línea de ajuste
        x_line = np.linspace(0.95, 1.05, 100)
        y_line = A * x_line + B
        fig.add_scatter(x=x_line, y=y_line, mode="lines", line=dict(color="red", width=2), name="Fit")

        # Línea referencia y = x
        fig.add_shape(type="line", x0=0.95, y0=0.95, x1=1.05, y1=1.05,
                      line=dict(color="gray", dash="dash"))
        fig.update_layout(showlegend=False)
        #fig.update_layout(coloraxis_colorbar=dict(title="Presence Index"))

    else:
        corr_text = "No hay suficientes datos para calcular correlación y ajuste"

    #fig.update_layout(width=width, height=height)
    fig.update_layout(
    title="Comparison of Normalized pYIN and ENF Frequencies for Playback Speed Inference",
    title_font_size=40,
    title_x=0.5,
    xaxis_title="pYIN / 440",
    yaxis_title="ENF / 50",
    xaxis_title_font_size=32,
    yaxis_title_font_size=32,
    xaxis_tickfont_size=28,
    yaxis_tickfont_size=28,
    legend_title_font_size=28,
    legend_font_size=24,
    xaxis_range=[0.95, 1.05],
    yaxis_range=[0.9, 1.1],
    width=width,
    height=height
)    
    return fig, corr_text

if __name__ == '__main__':
    app.run(debug=True)
