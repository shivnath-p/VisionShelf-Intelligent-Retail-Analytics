import streamlit as st
import os
from detect import detect_objects
from PIL import Image
import cv2

st.set_page_config(page_title="Smart Retail Shelf Monitoring System", layout="centered")
st.title("🛒 Smart Retail Shelf Monitoring System")

uploaded_file = st.file_uploader("Upload a shelf image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    # Save the uploaded image temporarily
    temp_path = "temp.jpg"
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.image(temp_path, caption="Uploaded Shelf Image", use_container_width=True)

    st.subheader("Detected Products:")
    df, result = detect_objects(temp_path)

    if not df.empty:
        st.dataframe(df[["name", "confidence"]])
        st.success(f" Total Detected: {len(df)}")

        # Annotate and save the image with bounding boxes
        annotated_img = result.plot()  # BGR image array
        output_path = "output.jpg"
        cv2.imwrite(output_path, annotated_img)

        # Display the annotated image
        st.image(output_path, caption="Detected Products", use_container_width=True)
    else:
        st.warning(" No products detected. Try another image.")
