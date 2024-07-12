import streamlit as st
import pandas as pd
from utils import generate_text_from_image, config_model, extract_receipt_info
from get_receipt import detect_and_crop
from pymongo import MongoClient
import json
from dotenv import load_dotenv
import os
import shutil
import re

load_dotenv()
hf_token = os.getenv('HF_TOKEN')

# Initialize the model
model, processor, device = config_model()

# Load MongoDB client
client = MongoClient(os.getenv('MONGODB_URI'))
db = client["receipts_db"]
receipts_collection = db["receipts"]

# Create directories if they don't exist
if not os.path.exists("images"):
    os.makedirs("images")
if not os.path.exists("inference"):
    os.makedirs("inference")

# Function to extract numeric value from a string
def extract_numeric_value(value_str):
    match = re.search(r'[\d.]+', value_str)
    return float(match.group()) if match else 0.0

# Function to calculate total from line items
def calculate_total(line_items):
    total = 0
    for item in line_items:
        try:
            value = extract_numeric_value(item["value"])
            quantity = float(item["quantity purchased"])
            total += value * quantity
        except (ValueError, KeyError):
            continue
    return round(total, 2)

# Streamlit UI
st.sidebar.title("Navigation")
page = st.sidebar.selectbox("Go to", ["Receipt Processing Application", "Stored Receipts"])

if page == "Receipt Processing Application":
    st.title("Receipt Processing Application")

    uploaded_file = st.file_uploader("Upload a receipt image", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        image_path = os.path.join("images", uploaded_file.name)
        with open(image_path, "wb") as buffer:
            shutil.copyfileobj(uploaded_file, buffer)

        # Extract the portion of the image which contains the receipt
        target_class_name = "book"
        cropped_image_path = detect_and_crop(image_path=image_path, target_class_name=target_class_name)

        if cropped_image_path:
            col1, col2, col3 = st.columns([1, 2, 1])
            with col1:
                st.write("")
            with col2:
                st.image(cropped_image_path, caption="Cropped Receipt Image", use_column_width='auto')
            with col3:
                st.write("")

            # Generate text from the image
            extracted_text = generate_text_from_image(model=model, image_path=cropped_image_path, processor=processor, device=device)
            receipt_data = json.loads(json.dumps(extracted_text))

            # Extract receipt information
            final_data = extract_receipt_info(receipt_data)

            # Display and edit line items in a table format
            st.write("Items Purchased:")
            line_items = final_data.get("line_items", [])
            for item in line_items:
                item.pop("item_key", None)  # Remove item_key if present
                item["name"] = item.pop("item_name", "")
                item["value"] = item.pop("item_value", "")
                item["quantity purchased"] = item.pop("item_quantity", "")
            df = pd.DataFrame(line_items)
            edited_df = st.data_editor(df)

            # Edit the extracted information
            with st.form(key="edit_form"):
                store_name = st.text_input("Store Name", final_data["store_name"])
                date_time = st.text_input("Date & Time", final_data["date_time"])
                total = st.text_input("Total", final_data.get("total", ""))
                if total == "":
                    total = calculate_total(edited_df.to_dict(orient="records"))
                save_button = st.form_submit_button(label="Save")
                cancel_button = st.form_submit_button(label="Cancel")

                if save_button:
                    final_data["store_name"] = store_name
                    final_data["date_time"] = date_time
                    final_data["total"] = total
                    final_data["line_items"] = edited_df.to_dict(orient="records")

                    # Save the final data to MongoDB
                    receipt_id = receipts_collection.insert_one(final_data).inserted_id
                    # st.success(f"Data saved successfully with ID: {receipt_id}")
                    st.success("Data saved successfully")

                if cancel_button:
                    st.warning("Process canceled. The data was not saved.")
                    st.experimental_rerun()  # Reset the form

elif page == "Stored Receipts":
    st.title("Stored Receipts")
    receipts = list(receipts_collection.find())
    for receipt in receipts:
        receipt["_id"] = str(receipt["_id"])
        with st.expander(f"Store: {receipt['store_name']} | Date & Time: {receipt['date_time']} | Total: {receipt['total']}"):
            # st.write(f"Receipt ID: {receipt['_id']}")
            st.write("Store Name:", receipt["store_name"])
            st.write("Date & Time:", receipt["date_time"])
            st.write("Total:", receipt["total"])

            # Display line items in a table format
            st.write("Items Purchased:")
            line_items = receipt.get("line_items", [])
            for item in line_items:
                item.pop("item_key", None)  # Remove item_key if present
                item["name"] = item.pop("name", "")
                item["value"] = item.pop("value", "")
                item["quantity purchased"] = item.pop("quantity purchased", "")
            df = pd.DataFrame(line_items)
            st.dataframe(df)

# Running the app - "streamlit run src/app.py"
