import io
from itertools import product

from matplotlib.backends.backend_agg import RendererAgg
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import streamlit as st
from streamlit_drawable_canvas import st_canvas

import bayesian as bs


WIDTH = 400
HEIGHT = 400
TASK_TO_MODELS = {
    '-': ('-',),
    'Regression': (
        '-',
        'Bayesian Linear Regression',
        'Variational Bayesian Linear Regression'
    ),
    'Classification': (
        '-',
        'Bayesian Logistic Regression',
    )
}


def get_fig_ax():
    plt.tight_layout()
    fig, ax = plt.subplots(
        subplot_kw={'xlim': (-1, 1), 'ylim': (-1, 1)})
    ax.grid(alpha=0.5)
    ax.tick_params(
        axis='x', which='both', bottom=False, top=False,
        labelbottom=False)
    ax.tick_params(
        axis='y', which='both', left=False, right=False,
        labelleft=False)
    return fig, ax


def figure_to_img(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=180, bbox_inches='tight', pad_inches=0)
    buf.seek(0)
    return Image.open(buf)


@st.cache(allow_output_mutation=True)
def get_background():
    return [None]


@st.cache(allow_output_mutation=True)
def get_cache():
    return {'x': None, 'y': None, 'model': None}


def main_app():
    st.sidebar.info('Select a task and a model above.')
    st.title('Welcom to Bayesian model playground!')
    st.markdown(
        '**Select a task and a model from the dropdown on the left** '
        'to play around with Bayesian models interactively!')


def select_feature():
    st.sidebar.subheader('Feature')
    features = []
    if st.sidebar.checkbox('Bias', value=True):
        features.append(bs.preprocess.BiasFeature())
    if st.sidebar.checkbox('Gaussian', value=True):
        num_locs = st.sidebar.slider('Number of Gaussian kernels', 1, 100, 10)
        scale = st.sidebar.slider('Scale of Gaussian kernels', 0.01, 10., 1.)
        features.extend([
            bs.preprocess.GaussianFeature([loc], scale) for loc
            in np.linspace(-1, 1, 2 * num_locs + 1)[1::2]
        ])
    if st.sidebar.checkbox('Sigmoid', value=False):
        num_locs = st.sidebar.slider('Number of sigmoid kernels', 1, 100, 10)
        scale = st.sidebar.slider('Scale of sigmoid kernels', 0.01, 10., 1.)
        features.extend([
            bs.preprocess.SigmoidalFeature([loc], scale) for loc
            in np.linspace(-1, 1, 2 * num_locs + 1)[1::2]
        ])
    if st.sidebar.checkbox('Polynomial', value=False):
        degrees = st.sidebar.multiselect(
            'Degrees', [f'x^{i}' for i in range(1, 10)], default=['x^1'])
        features.extend([
            bs.preprocess.PolynomialFeature(int(deg[2])) for deg in degrees])
    return bs.preprocess.StackedFeatures(*features)


def select_feature_for_classification():
    st.sidebar.subheader('Feature')
    features = []
    if st.sidebar.checkbox('Bias', value=True):
        features.append(bs.preprocess.BiasFeature())
    if st.sidebar.checkbox('Gaussian', value=True):
        n = st.sidebar.slider('Number of Gaussian kernels per axis', 1, 20, 10)
        scale = st.sidebar.slider('Scale of Gaussian kernels', 0.01, 10., 1.)
        features.extend([
            bs.preprocess.GaussianFeature([x1, x2], scale) for x1, x2
            in product(
                np.linspace(-1, 1, 2 * n + 1)[1::2],
                np.linspace(-1, 1, 2 * n + 1)[1::2])
        ])
    if st.sidebar.checkbox('Polynomial', value=False):
        degrees = st.sidebar.multiselect(
            'Degrees', [f'x^{i}' for i in range(1, 10)], default=['x^1'])
        features.extend([
            bs.preprocess.PolynomialFeature(int(deg[2])) for deg in degrees])
    if len(features) == 0:
        st.sidebar.error('Select at least one feature')
    return bs.preprocess.StackedFeatures(*features)


def bayesian_linear_regression():
    st.markdown(bs.linear.Regression.__doc__)
    feature = select_feature()
    st.sidebar.subheader('Hyperparameters')
    alphas = np.logspace(-5, 0, 100).tolist()
    betas = np.logspace(-1, 4, 100).tolist()
    return bs.linear.Regression(
        alpha=st.sidebar.select_slider('alpha', alphas, alphas[50]),
        beta=st.sidebar.select_slider('beta', betas, betas[50]),
        feature=feature)


def variational_bayesian_linear_regression():
    st.markdown(bs.linear.VariationalRegression.__doc__)
    feature = select_feature()
    st.sidebar.subheader('Hyperparameters')
    a0s = np.logspace(-3, 2, 100).tolist()
    b0s = np.logspace(-3, 2, 100).tolist()
    betas = np.logspace(-1, 4, 100).tolist()
    return bs.linear.VariationalRegression(
        a0=st.sidebar.select_slider('a0', a0s, a0s[50]),
        b0=st.sidebar.select_slider('b0', b0s, b0s[50]),
        beta=st.sidebar.select_slider('beta', betas, betas[50]),
        feature=feature)


def bayesian_logistic_regression():
    st.markdown(bs.linear.Classifier.__doc__)
    feature = select_feature_for_classification()
    alphas = np.logspace(-5, 0, 100).tolist()
    return bs.linear.Classifier(
        alpha=st.sidebar.select_slider('alpha', alphas, alphas[50]),
        feature=feature)


def regression_app(model: str):
    action = st.sidebar.selectbox('Action:', (
        'Click on the canvas to add data points',
        'Double click points to delete them',))
    try:
        model = eval(model.lower().replace(' ', '_'))()
    except NameError:
        st.error('Not Implemeted Error')
        st.stop()

    placeholder = st.empty()
    with placeholder.beta_container():
        bg = get_background()
        if bg[0] is None:
            with RendererAgg.lock:
                fig, ax = get_fig_ax()
                bg[0] = figure_to_img(fig)
                plt.clf()
        canvas = st_canvas(
            stroke_width=10, stroke_color='blue', background_image=bg[0],
            drawing_mode='circle' if 'add' in action else 'transform',
            update_streamlit=True, width=WIDTH, height=HEIGHT, key='canvas')
    dataset = [
        obj for obj in canvas.json_data['objects'] if obj['type'] == 'circle']
    x_train = [2 * obj['left'] / WIDTH - 1 for obj in dataset]
    y_train = [1 - 2 * obj['top'] / HEIGHT for obj in dataset]
    cache = get_cache()
    if ((cache['model'] != model)
            or (cache['x'] != x_train)
            or (cache['y'] != y_train)):
        cache['x'] = x_train
        cache['y'] = y_train
        cache['model'] = model
        with RendererAgg.lock:
            fig, ax = get_fig_ax()
            if len(dataset) > 0:
                model.fit(x_train, y_train)
                cache['model'] = model
                x = np.linspace(-1, 1, 100)
                y, y_std = model.predict(x)
                ax.plot(x, y, c='C1')
                ax.fill_between(x, y - y_std, y + y_std, color='C1', alpha=0.2)
            bg[0] = figure_to_img(fig)
            plt.clf()
        st.experimental_rerun()


def classification_app(model: str):
    action = st.sidebar.selectbox('Action:', (
        'Click to add positive data points',
        'Click to add negative data points',
        'Double click to delete data points',
    ))
    try:
        func = eval(model.lower().replace(' ', '_'))
    except NameError:
        st.error('Not Implemented Error')
        st.stop()
    model = func()
    bg = get_background()
    if bg[0] is None:
        with RendererAgg.lock:
            fig, ax = get_fig_ax()
            bg[0] = figure_to_img(fig)
            plt.clf()
    canvas = st_canvas(
        stroke_width=10,
        stroke_color='yellow' if 'positive' in action else 'purple',
        background_image=bg[0],
        drawing_mode='circle' if 'add' in action else 'transform',
        update_streamlit=True, width=WIDTH, height=HEIGHT, key='canvas')
    dataset = [
        obj for obj in canvas.json_data['objects'] if obj['type'] == 'circle']
    x_train = [
        [2 * obj['left'] / WIDTH - 1, 1 - 2 * obj['top'] / HEIGHT]
        for obj in dataset]
    y_train = [int(obj['stroke'] == 'yellow') for obj in dataset]
    cache = get_cache()
    if ((cache['model'] != model)
            or (cache['x'] != x_train)
            or (cache['y'] != y_train)):
        cache['x'] = x_train
        cache['y'] = y_train
        cache['model'] = model
        with RendererAgg.lock:
            fig, ax = get_fig_ax()
            if len(dataset) > 0:
                model.fit(x_train, y_train)
                cache['model'] = model
                x = np.linspace(-1, 1, 100)
                x1, x2 = np.meshgrid(x, x)
                x = np.array([x1, x2]).reshape(2, -1).T
                y = model.proba(x)
                ax.contourf(x1, x2, y.reshape(100, 100), alpha=0.2)
            bg[0] = figure_to_img(fig)
            plt.clf()
        st.experimental_rerun()


if __name__ == "__main__":
    task = st.sidebar.selectbox('Select a task:', tuple(TASK_TO_MODELS.keys()))
    model = st.sidebar.selectbox('Select a model:', TASK_TO_MODELS[task])
    if model == '-':
        main_app()
    else:
        eval(task.lower() + '_app')(model)
