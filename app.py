import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from streamlit_drawable_canvas import st_canvas

import bayesian as bs


TASK_TO_MODELS = {
    '-': ('-',),
    'Regression': ('-', 'Bayesian Linear Regression',),
}


def main_app():
    st.sidebar.info('Select a task and a model above.')
    st.title('Welcom to Bayesian model playground!')
    st.markdown(
        '**Select a task and a model from the dropdown on the left** '
        'to play around with Bayesian models interactively!')


def select_feature():
    st.sidebar.subheader('Feature')
    feature = st.sidebar.selectbox('Feature Type:', (
        'Gaussian', 'Polynomial', 'Sigmoid'))
    if feature == 'Polynomial':
        use_nth_deg = [
            st.sidebar.checkbox(f'x^{i}', value=True) for i in range(10)]
        if any(use_nth_deg) is False:
            st.sidebar.error('Please choose at least one feature above.')
        return bs.preprocess.PolynomialFeatures(use_nth_deg)
    elif feature == 'Gaussian':
        num_locs = st.sidebar.slider('Number of Gaussian kernels', 1, 100, 10)
        scale = st.sidebar.slider('Scale of Gaussian kernels', 0.01, 10., 1.)
        return bs.preprocess.GaussianFeatures(
            np.linspace(-1, 1, 2 * num_locs + 1)[1::2], scale)
    elif feature == 'Sigmoid':
        num_locs = st.sidebar.slider('Number of sigmoid kernels', 1, 100, 10)
        scale = st.sidebar.slider('Scale of sigmoid kernels', 0.01, 10., 1.)
        return bs.preprocess.SigmoidalFeature(
            np.linspace(-1, 1, 2 * num_locs + 1)[1::2], scale)


def regression_app(model: str):
    action = st.sidebar.selectbox('Action:', (
        'Click on the canvas to add data points',
        'Double click points to delete them',))
    feature = select_feature()
    if model == 'Bayesian Linear Regression':
        cls_ = bs.linear.Regression
        st.sidebar.subheader('Hyperparameters')
        alphas = np.logspace(-5, 0, 100).tolist()
        betas = np.logspace(-1, 4, 100).tolist()
        model = cls_(
            feature,
            st.sidebar.select_slider('Alpha', alphas, alphas[50]),
            st.sidebar.select_slider('Beta', betas, betas[50]))
    st.markdown(cls_.__doc__)
    width = 600
    height = 400
    canvas = st_canvas(
        stroke_width=10,
        stroke_color='blue',
        background_color='lightgray',
        update_streamlit=True,
        drawing_mode='circle' if 'add' in action else 'transform',
        width=width,
        height=height,
        key='canvas',
    )
    circles = [
        obj for obj in canvas.json_data['objects'] if obj['type'] == 'circle']
    if len(circles) > 0:
        x_train = [2 * obj['left'] / width - 1 for obj in circles]
        y_train = [1 - 2 * obj['top'] / height for obj in circles]
        fig, ax = plt.subplots(subplot_kw={'xlim': (-1, 1), 'ylim': (-1, 1)})
        model.fit(x_train, y_train)
        x = np.linspace(-1, 1, 100)
        y, y_std = model.predict(x)
        ax.plot(x, y, c='C1')
        ax.fill_between(x, y - y_std, y + y_std, color='C1', alpha=0.2)
        ax.scatter(x_train, y_train, c='C0')
        ax.grid(alpha=0.5)
        st.pyplot(fig)


if __name__ == "__main__":
    task = st.sidebar.selectbox('Select a task:', tuple(TASK_TO_MODELS.keys()))
    model = st.sidebar.selectbox('Select a model:', TASK_TO_MODELS[task])
    if model == '-':
        main_app()
    elif task == 'Regression':
        regression_app(model)
