import streamlit as st

st.set_page_config(
    page_title="ReturnWise",
    page_icon="site/assets/returnwise-mark.svg",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.logo(
    "site/assets/returnwise-wordmark.svg",
    icon_image="site/assets/returnwise-mark.svg",
    size="large",
)

page = st.navigation(
    {
        "Workspace": [
            st.Page("app_pages/01_overview.py", title="Overview", icon=":material/dashboard:", default=True),
            st.Page("app_pages/02_data.py", title="Dataset and quality", icon=":material/database:"),
            st.Page("app_pages/03_analysis.py", title="EDA and statistics", icon=":material/query_stats:"),
        ],
        "Decision support": [
            st.Page("app_pages/04_risk_queue.py", title="Risk queue", icon=":material/low_priority:"),
            st.Page("app_pages/05_decision_simulator.py", title="Decision simulator", icon=":material/tune:"),
            st.Page("app_pages/06_research.py", title="Models and research", icon=":material/science:")
        ],
    }
)
page.run()
