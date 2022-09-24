import os
from typing import Any, Optional, SupportsInt

import numpy as np
import pandas as pd
import streamlit as st
from plotly import graph_objects as go
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (
    DotProduct as DP,
    Kernel as K,
    Matern as M,
    WhiteKernel as W,
)

st.set_page_config(layout="wide")


def R2(Y_true: np.ndarray, Y_pred: np.ndarray) -> np.float64:
    Y_true = np.array(Y_true, dtype=np.float64)
    Y_pred = np.array(Y_pred, dtype=np.float64)
    S_tot = np.sum((Y_true - np.mean(Y_true)) ** 2)
    S_res = np.sum((Y_true - Y_pred) ** 2)
    return 1 - S_res / S_tot


def RGB2Hex(R, G, B) -> str:
    return f"#{int(R):02x}{int(G):02x}{int(B):02x}"


def Hex2RGB(hex: str) -> tuple[int, int, int]:
    return tuple(int(hex.strip("#")[i : i + 2], 16) for i in (0, 2, 4))


def grey_scale(r: SupportsInt, g: SupportsInt, b: SupportsInt) -> int:
    return round(int(r) * 0.299 + int(g) * 0.587 + int(b) * 0.114)


def parse_data_frame(df: pd.DataFrame) -> pd.DataFrame:
    df["logC"] = np.log10(df["C"], where=df["C"] != 0)
    df.loc[df["C"] == 0, "logC"] = -np.inf

    df["logC1"] = np.log10(df["C"] + 1)

    df["Grey"] = df.apply(lambda x: grey_scale(x["R"], x["G"], x["B"]), axis=1)
    return df


@st.cache(allow_output_mutation=True)
def fit_model(X, Y, kernel: K = 1 * W() + 1 * DP()):
    model = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10)
    model.fit(X, Y)
    return model


def plot3D(
    name: str,
    x,
    y,
    z,
    z_mean,
    z_std=Any,
    CI: int = 95,
    x_sample=None,
    y_sample=None,
    z_sample_mean=None,
    z_sample_std=None,
):
    alpha = (1 - CI / 100) / 2

    fig = go.Figure(layout=plotly_layout)
    if z_std is not None:
        z_upper = z_mean + stats.norm.ppf(1 - alpha) * z_std
        z_lower = z_mean - stats.norm.ppf(1 - alpha) * z_std
        fig.add_trace(
            go.Scatter3d(
                x=x,
                y=y,
                z=z_lower,
                mode="lines",
                name=f"{CI}% CI lb",
                opacity=0.5,
                line_color="lightskyblue",
            ),
        )
        fig.add_trace(
            go.Scatter3d(
                x=x,
                y=y,
                z=z_upper,
                mode="lines",
                name=f"{CI}% CI ub",
                opacity=0.5,
                line_color="lightskyblue",
            ),
        )
    fig.add_trace(
        go.Scatter3d(
            x=x,
            y=y,
            z=z_mean,
            mode="lines",
            name=f"Predicted mean (R^2={R2(z, z_mean):.3f})",
            line_color="skyblue",
        )
    )
    fig.add_trace(
        go.Scatter3d(
            x=x,
            y=y,
            z=z,
            mode="markers",
            name="Observations",
            marker=dict(size=3, color="crimson"),
        ),
    )
    if (
        x_sample is not None
        and y_sample is not None
        and z_sample_mean is not None
        and z_sample_std is not None
    ):
        fig.add_trace(
            go.Scatter3d(
                x=x_sample,
                y=y_sample,
                z=z_sample_mean,
                mode="markers",
                name=f"Prediction (1σ) {float(z_sample_mean):.4f}±{float(z_sample_std):.4f}",
                marker=dict(size=3, color="darkorange"),
            )
        )
    fig.update_layout(
        legend_title=name,
        scene_xaxis_showticklabels=False,
        scene_yaxis_showticklabels=False,
        scene_zaxis_showticklabels=False,
        scene=dict(
            xaxis_title="Color PC1",
            yaxis_title="Color PC2",
            zaxis_title="log(C+1)",
            xaxis=dict(
                gridcolor="darkgrey",
                showbackground=False,
                zerolinecolor="white",
            ),
            yaxis=dict(
                gridcolor="darkgrey",
                showbackground=False,
                zerolinecolor="white",
            ),
            zaxis=dict(
                gridcolor="darkgrey",
                showbackground=False,
                zerolinecolor="white",
            ),
        ),
    )
    return fig


def plot3DVolume(
    name: str,
    R,
    G,
    B,
    S,
    R_sample=None,
    G_sample=None,
    B_sample=None,
    S_sample_mean=None,
    S_sample_std=None,
):
    fig = go.Figure(layout=plotly_layout)
    fig.add_trace(
        go.Volume(
            x=R.flatten(),
            y=G.flatten(),
            z=B.flatten(),
            value=S.flatten(),
            opacity=0.5,
            colorscale="viridis",
            name="Prediction over RGB color space",
            showlegend=True,
        )
    )
    if (
        R_sample is not None
        and G_sample is not None
        and B_sample is not None
        and S_sample_mean is not None
        and S_sample_std is not None
    ):
        fig.add_trace(
            go.Scatter3d(
                x=R_sample,
                y=G_sample,
                z=B_sample,
                mode="markers",
                name=f"Prediction (1σ) {float(S_sample_mean):.4f}±{float(S_sample_std):.4f}",
                marker=dict(
                    size=5,
                    color=S_sample_mean,
                    cmin=np.min(S),
                    cmax=np.max(S),
                    colorscale="viridis",
                ),
            )
        )
    fig.update_layout(
        legend_title=name,
        scene_xaxis_showticklabels=False,
        scene_yaxis_showticklabels=False,
        scene_zaxis_showticklabels=False,
        scene=dict(
            xaxis_title="R",
            yaxis_title="G",
            zaxis_title="B",
            xaxis=dict(
                gridcolor="darkgrey",
                showbackground=False,
                zerolinecolor="white",
            ),
            yaxis=dict(
                gridcolor="darkgrey",
                showbackground=False,
                zerolinecolor="white",
            ),
            zaxis=dict(
                gridcolor="darkgrey",
                showbackground=False,
                zerolinecolor="white",
            ),
        ),
    )
    return fig


plotly_layout = go.Layout(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    xaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.5)"),
    yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.25)"),
    margin=go.layout.Margin(
        l=0,
        r=0,
        b=0,
        t=0,
    ),
    legend=dict(
        orientation="h",
        xanchor="center",
        yanchor="top",
        x=0.5,
        y=-0.05,
    ),
)

data_root = "./data"
os.chdir(os.path.dirname(os.path.abspath(__file__)))
file_list = [
    os.path.join(data_root, file)
    for file in os.listdir(data_root)
    if file.endswith(".csv")
]
file_list.sort()
file_name_list = [os.path.basename(file).strip(".csv") for file in file_list]

file_name: str = st.selectbox("Select a experiment type", file_name_list)
file_name = os.path.join(data_root, f"{file_name}.csv")
df = pd.read_csv(file_name)

CI: int = st.slider("CI", 0, 100, 95)
col_L, col_M, col_R = st.columns(3)
with col_L:
    C = st.color_picker(
        "Pick a color", RGB2Hex(df["R"].mean(), df["G"].mean(), df["B"].mean())
    )

with st.expander("Raw data"):
    st.dataframe(parse_data_frame(df), use_container_width=True)

kernel: K = 1 * W() + 1 * DP()

X, Y = df[["R", "G", "B"]].values, df["logC1"].values
R, G, B = X.T

RA, GA, BA = np.meshgrid(
    np.arange(256, step=4, dtype=np.float64),
    np.arange(256, step=4, dtype=np.float64),
    np.arange(256, step=4, dtype=np.float64),
)

with st.expander("Gaussian process regression"):
    gp = fit_model(X, Y, kernel)
    Y_p_mean, Y_p_std = gp.predict(X, return_std=True)
    pca = PCA(n_components=3)
    X_pca = pca.fit_transform(X)
    X_sample = np.array(Hex2RGB(C)).reshape(1, -1)
    X_sample_pca = pca.transform(X_sample)
    Y_sample_mean, Y_sample_std = gp.predict(X_sample, return_std=True)

    fig = plot3D(
        "GPR",
        X_pca[:, 0],
        X_pca[:, 1],
        Y,
        Y_p_mean,
        Y_p_std,
        CI,
        X_sample_pca[:, 0],
        X_sample_pca[:, 1],
        Y_sample_mean,
        Y_sample_std,
    )
    st.plotly_chart(fig, use_container_width=True)

    Y_p_mean, Y_p_std = gp.predict(
        np.stack((RA.flatten(), GA.flatten(), BA.flatten()), axis=1), return_std=True
    )
    fig = plot3DVolume(
        "GPR",
        RA,
        GA,
        BA,
        Y_p_mean,
        X_sample[:, 0],
        X_sample[:, 1],
        X_sample[:, 2],
        Y_sample_mean,
        Y_sample_std,
    )
    st.plotly_chart(fig, use_container_width=True)

with col_M:
    st.metric(
        "Predicted concentration in log10:",
        f"{float(Y_sample_mean):.4f}±{float(Y_sample_std):.4f}",
        None,
        "off",
    )

with col_R:
    st.metric(
        "Predicted concentration:",
        f"{10**(float(Y_sample_mean)-float(Y_sample_std))-1:.4e}"
        f"~{10**(float(Y_sample_mean)+float(Y_sample_std))-1:.4e}",
        None,
        "off",
    )
