import streamlit as st
import requests
import time

st.set_page_config(page_title="Admin Panel", layout="wide")

# --------- LOGIN SYSTEM ----------
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

def login_page():
    st.title("🔐 Admin Login")

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        # Demo credentials (change later)
        if username == "admin" and password == "1234":
            st.session_state.logged_in = True
            st.success("✅ Login Successful")
            st.rerun()
        else:
            st.error("❌ Invalid username or password")

def dashboard_page():
    st.title("👥 AI Crowd Counting System - Admin Dashboard")

    if st.button("🚪 Logout"):
        st.session_state.logged_in = False
        st.rerun()

    col1, col2, col3 = st.columns(3)

    entered_box = col1.empty()
    exited_box = col2.empty()
    inside_box = col3.empty()

    status = st.empty()

    while True:
        try:
            r = requests.get("http://localhost:5005/api/live", timeout=2)
            data = r.json()

            entered = int(data.get("entered", 0))
            exited = int(data.get("exited", 0))
            inside = entered - exited

            entered_box.metric("Total Entered", entered)
            exited_box.metric("Total Exited", exited)
            inside_box.metric("Currently Inside", inside)

            status.success("✅ System Live")

        except:
            status.error("❌ Admin Server Not Running")

        time.sleep(1)

# --------- ROUTER ----------
if st.session_state.logged_in:
    dashboard_page()
else:
    login_page()