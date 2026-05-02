import streamlit as st
import pickle

# Page config
st.set_page_config(page_title="Student Performance Analyzer", layout="centered")

# Load model
model = pickle.load(open("model.pkl", "rb"))

# Title
st.title("🎓 AI Student Performance Analyzer")
st.markdown("### Predict student marks based on performance 📊")

st.divider()

# Layout using columns
col1, col2 = st.columns(2)

with col1:
    hours = st.slider("📚 Study Hours", 0, 10)

with col2:
    attendance = st.slider("📅 Attendance (%)", 0, 100)

previous = st.slider("📝 Previous Marks", 0, 100)

st.divider()

# Prediction button
if st.button("🚀 Predict Performance"):
    prediction = model.predict([[hours, attendance, previous]])

    st.subheader("📊 Result")
    st.success(f"Predicted Final Marks: {prediction[0]:.2f}")

    st.progress(min(int(prediction[0]), 100))

    # Feedback
    if prediction[0] < 60:
        st.warning("⚠️ Needs improvement. Try increasing study hours.")
    else:
        st.success("✅ Good performance expected!")

    # Extra tip
    if hours < 3:
        st.info("💡 Tip: Studying more than 3 hours daily can improve results.")