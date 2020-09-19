import streamlit as st


def main_app():
    st.sidebar.info('Select a model above.')
    st.title('Welcom to Bayesian model playground!')
    st.markdown(
        '**Select a model from the dropdown on the left** to play around with '
        'Bayesian models interactively!')


if __name__ == "__main__":
    model = st.sidebar.selectbox('Select a model you want to play around:', (
        '-',
    ))
    if model == '-':
        main_app()
