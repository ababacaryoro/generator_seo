import base64
from typing import Optional

import streamlit as st


def streamlit_custom_page(
    set_page_config: Optional[bool] = True,
    logo_path: str = None,
    css_path: str = None,
    background_image_path: str = None,
    page_title="",
) -> None:
    """Render the UI streamlit template.

    Args:
        set_page_config (bool, optional): Sets the streamlit page config. Defaults to True.
        logo_path: path to the logo image. Defaults to None.
    """
    if set_page_config:
        st.set_page_config(
            page_title=page_title,
            page_icon=logo_path,
            layout="wide",
            initial_sidebar_state="collapsed",
        )

    # CoE logo on the sidebar
    st.sidebar.image(logo_path)
    st.sidebar.markdown("<br>", unsafe_allow_html=True)

    # Set custom PR css template
    def local_css(file_name):
        with open(file_name) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

    # Background image
    @st.cache_data
    def get_base64_of_bin_file(bin_file):
        with open(bin_file, "rb") as f:
            data = f.read()
        return base64.b64encode(data).decode()

    def set_png_as_page_bg(png_file):
        bin_str = get_base64_of_bin_file(png_file)
        page_bg_img = (
            """
        <style>
        .stApp {
        background-image: url("data:image/png;base64,%s") !important;
        background-size: cover;
        }
        </style>
        """
            % bin_str
        )

        st.markdown(page_bg_img, unsafe_allow_html=True)
        return

    if background_image_path is not None:
        set_png_as_page_bg(background_image_path)

    # Call the css template
    local_css(css_path)
