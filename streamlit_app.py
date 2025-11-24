import streamlit as st
import pandas as pd
from components import agent_query, generate_viz


def main():
    with open("styles.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

    # Initialize session state for file bytes
    if 'file_bytes' not in st.session_state:
        sample_data_path = "data/sample_data.json"
        with open(sample_data_path, "rb") as f:
            st.session_state.file_bytes = f.read()  # Read the bytes, not the file handle

    if 'user_input' not in st.session_state:
        st.session_state.user_input = ""

    st.title("🤖 HSE Bot - Visualization Agent")
    st.markdown(
        "I am the Visualization Agent for the HSE Bot. Give me the user prompt and the data, and I will visualize it for you.")

    col1, col2 = st.columns([3, 1], vertical_alignment="center")
    with col1:
        st.markdown(
            """
            Voit kysyä myös suomeksi!
            <img src="https://upload.wikimedia.org/wikipedia/commons/b/bc/Flag_of_Finland.svg" alt="Finnish Flag" width="20"/>
            """,
            unsafe_allow_html=True)

    with col2:
        with st.container(vertical_alignment="bottom"):
            if st.download_button(
                label="Try with sample data",
                use_container_width=True,
                data=st.session_state.file_bytes,
                file_name="sample.json",
                mime="application/json"
            ):
                # Update prompt when button is clicked
                st.session_state.user_input = "Analyze the safety observations in this dataset"
                st.rerun()

    user_input = st.text_area(
        "Enter the user prompt",
        key="user_input"
    )

    user_input_file = st.file_uploader(
        "Upload the dataframe file",
        type=["csv", "xlsx", "json"]
    )

    if st.button("Get Visualization"):
        if user_input.strip() and user_input_file:
            try:
                if user_input_file.name.endswith('.csv'):
                    df = pd.read_csv(user_input_file)
                elif user_input_file.name.endswith('.json'):
                    df = pd.read_json(user_input_file)
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
