import streamlit as st
import pandas as pd
from components import agent_query, generate_viz


def main():
    with open("styles.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

    # st.session_state.filters = {}
    # st.session_state.hard_limit = 300

    st.title("🤖 HSE Bot - Visualization Agent")
    st.markdown(
        "I am the Visualization Agent for the HSE Bot. Give me the user prompt and the data, and I will visualize it for you.")
    st.markdown(
        """
        Voit kysyä myös suomeksi!
        <img src="https://upload.wikimedia.org/wikipedia/commons/b/bc/Flag_of_Finland.svg" alt="Finnish Flag" width="20"/>
        """,
        unsafe_allow_html=True)

    user_input = st.text_area("Enter the user prompt", key="user_input")
    user_input_file = st.file_uploader("Upload the dataframe file", type=[
                                       "csv", "xlsx"], key="user_file_upload")

    if st.button("Get Visualization"):
        if user_input.strip() and user_input_file:
            try:
                if user_input_file.name.endswith('.csv'):
                    df = pd.read_csv(user_input_file)
                else:
                    df = pd.read_excel(user_input_file)
            except Exception as e:
                st.error(f"Error reading file: {e}")
                return
            try:
                res = agent_query(user_input.strip(), df)
                generate_viz(*res)

            except Exception as e:
                st.error(f"Error getting visualization: {e}")
        else:
            st.warning("Please enter valid user prompt and the queried data.")


if __name__ == "__main__":
    main()
