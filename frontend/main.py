import os
import numpy as np
import streamlit as st
import pandas as pd
from dotenv import load_dotenv
from backend_requests import get_visualization
from chart_factory import make_chart


def main():

    with open("styles.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

    # st.session_state.filters = {}
    # st.session_state.hard_limit = 300

    st.title("🤖 HSE Bot - Visualization Agent")
    st.markdown("I am the Visualization Agent for the HSE Bot. Give me the user prompt and the data, and I will visualize it for you.")
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
                get_visualization(user_input, df) 
            except Exception as e:
                st.error(f"Error getting visualization: {e}")
        else:
            st.warning("Please enter valid user prompt and the queried data.")
            
        
     
    chart_data = pd.DataFrame(
        np.random.randn(20, 3),
        columns=['Series A', 'Series B', 'Series C']
    )
    
    st.subheader("📊 Generated Visualization")
    line_chart = make_chart(chart_data, {
        "chart_type": "line",
        "channels": {
            "x": "Series A",
            "y": "Series B",
        }
    }).get_chart()
    
    st.altair_chart(line_chart)


if __name__ == "__main__":

    load_dotenv()
    # Set some env here
    main()
