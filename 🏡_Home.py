import streamlit as st

st.set_page_config(page_title="PaperProbe", page_icon=":card_index_dividers:", layout="wide")

st.title(":card_index_dividers: PaperProbe \n\n **Smart Study Companion**")

#st.balloons()

with open("docs/intro.md", "r") as f:
    st.success(f.read())

with open("docs/features.md", "r") as f:
    st.info(f.read())

