from io import StringIO
import streamlit as st
import pandas as pd
from dotenv import load_dotenv
from agent_caller import get_response, get_test_response
from chart_factory import make_chart


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
                with st.spinner('Agents are working...', show_time=True):
                    buffer = StringIO()
                    buffer.write("Sending user prompt and data to agents...\n")
                    
                    with st.expander("📝 Log", expanded=False):
                        placeholder = st.empty()
                        placeholder.code(buffer.getvalue(), language="text")
                        
                    def update_output(output: str):
                        buffer.write(output + "\n")
                        placeholder.code(buffer.getvalue(), language="text")
                    
                    df, chart_spec = get_response(user_input, df, output_callback=update_output)
                
                st.subheader("📊 Generated Visualization")
                
                line_chart = make_chart(df, chart_spec)
                st.plotly_chart(line_chart)
                
            except Exception as e:
                st.error(f"Error getting visualization: {e}")
        else:
            st.warning("Please enter valid user prompt and the queried data.")
            
        
    df, chart_spec = get_test_response()
    st.subheader("📊 Generated Visualization")
    line_chart = make_chart(df, chart_spec)
    st.plotly_chart(line_chart)
    with st.expander("📋 Details", expanded=False):
        tab1, tab2 = st.tabs(["Preprocessing Steps", "Rationale"])
        
        with tab1:
            st.markdown("### Data Preprocessing")
            st.write("The data was processed to prepare it for visualization.")
        
        with tab2:
            st.markdown("### Visualization Rationale")
            st.write("The chart was generated based on the user prompt and data analysis.")


if __name__ == "__main__":
    load_dotenv()
    main()
