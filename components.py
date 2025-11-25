import pandas as pd
import streamlit as st
from io import StringIO
from chart_factory import make_chart
from agent_caller import get_response


def agent_query(user_input: str, df: pd.DataFrame):
    with st.spinner('Agents are working...', show_time=True):
        buffer = StringIO()
        buffer.write("Sending user prompt and data to agents...\n")

        with st.expander("📝 Log", expanded=False):
            with st.container(border=False, height=500, horizontal_alignment="center"):
                placeholder = st.empty()
                placeholder.code(buffer.getvalue(), language="text")

        def update_output(output: str):
            buffer.write(output + "\n")
            placeholder.code(buffer.getvalue(), language="text")

        res = get_response(user_input, df, output_callback=update_output)
    return res


def generate_viz(df, chart_metadata, preprocessing_steps, rationale):
    subheader_placeholder = st.empty()
    
    with st.spinner('Generating Visualization...', show_time=True):
        chart = make_chart(df, chart_metadata)
        
        with st.container(border=True, horizontal_alignment="center", vertical_alignment="center"):
            st.plotly_chart(chart)
            
        subheader_placeholder.subheader("📊 Generated Visualization")

        with st.expander("📋 Details", expanded=False):
            with st.container(border=False, height=400):
                tab1, tab2 = st.tabs(["Preprocessing Steps", "Rationale"])

                with tab1:
                    st.markdown("### Data Preprocessing Steps")
                    for index, step in enumerate(preprocessing_steps, 1):
                        st.markdown(f"{index}. {step}")

                with tab2:
                    st.markdown("### Visualization Rationale")
                    st.write(rationale)
