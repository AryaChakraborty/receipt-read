import streamlit as st
import pandas as pd
from utils import generate_text_from_image, config_model, extract_receipt_info
from get_receipt import detect_and_crop
from analyze_csv import analyze_data
from pymongo import MongoClient
import json
from dotenv import load_dotenv
import os
import shutil
import re
import cloudinary
import cloudinary.uploader
import cloudinary.api

# Load environment variables
load_dotenv()
hf_token = os.getenv('HF_TOKEN')
mongo_uri = os.getenv('MONGODB_URI')
bamboollm_api_key = os.getenv('BAMBOO_LLM_API_KEY')
cloudinary.config(
    cloud_name=os.getenv('CLOUDINARY_CLOUD_NAME'),
    api_key=os.getenv('CLOUDINARY_API_KEY'),
    api_secret=os.getenv('CLOUDINARY_API_SECRET')
)

# Initialize the model
model, processor, device = config_model()

# Load MongoDB client
client = MongoClient(mongo_uri)
db = client["receipts_db"]
receipts_collection = db["receipts"]
folders_collection = db["folders"]

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

# Function to upload image to Cloudinary and get the URL
def upload_image_to_cloudinary(image_path):
    response = cloudinary.uploader.upload(image_path)
    return response['secure_url']

def update_csv_file():

    # Ensure the csv_data folder exists
    if not os.path.exists("csv_data"):
        os.makedirs("csv_data")

    # Retrieve all receipts from MongoDB
    receipts = list(receipts_collection.find())

    # Initialize a list to store line items
    line_items_list = []

    # Extract relevant data for each line item
    for receipt in receipts:
        store_name = receipt["store_name"]
        folder_name = receipt["folder_name"]
        date_time = receipt["date_time"]
        line_items = receipt.get("line_items", [])

        for item in line_items:
            line_items_list.append({
                "store name": store_name,
                "folder name": folder_name,
                "date time": date_time,
                "name": item.get("name", ""),
                "price": item.get("value", ""),
                "quantity purchased": item.get("quantity purchased", "")
            })

    # Create a DataFrame from the line items list
    df = pd.DataFrame(line_items_list)

    # Reorder the columns
    column_order = ["name", "price", "folder name", "store name", "quantity purchased", "date time"]
    df = df[column_order]

    # Save the DataFrame to a CSV file
    csv_file_path = os.path.join("csv_data", "receipts_data.csv")
    df.to_csv(csv_file_path, index=False)

    print(f"CSV file updated and saved to {csv_file_path}")

# Streamlit UI
st.sidebar.title("Navigation")
page = st.sidebar.selectbox("Go to", ["Receipt Processing Application", "Stored Receipts", "Analyze"])

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

            # Get list of folder names for dropdown
            folders = folders_collection.find()
            folder_names = [folder['folder_name'] for folder in folders]

            # Edit the extracted information
            with st.form(key="edit_form"):
                store_name = st.text_input("Store Name", final_data["store_name"])
                date_time = st.text_input("Date & Time", final_data["date_time"])
                total = st.text_input("Total", final_data.get("total", ""))
                if total == "":
                    total = calculate_total(edited_df.to_dict(orient="records"))
                folder_name = st.selectbox("Select Folder", folder_names)
                save_button = st.form_submit_button(label="Save")
                cancel_button = st.form_submit_button(label="Cancel")

                if save_button:
                    final_data["store_name"] = store_name
                    final_data["date_time"] = date_time
                    final_data["total"] = total
                    final_data["line_items"] = edited_df.to_dict(orient="records")
                    final_data["folder_name"] = folder_name

                    # Upload receipt image to Cloudinary and save the URL
                    image_url = upload_image_to_cloudinary(cropped_image_path)
                    final_data["image_url"] = image_url

                    # Save the final data to MongoDB
                    receipt_id = receipts_collection.insert_one(final_data).inserted_id

                    # Save the folder name if not already exists
                    if not folders_collection.find_one({"folder_name": folder_name}):
                        folders_collection.insert_one({"folder_name": folder_name})

                    st.success("Data saved successfully")

                    # once data is saved, update the csv file
                    update_csv_file()

                if cancel_button:
                    st.warning("Process canceled. The data was not saved.")
                    st.experimental_rerun()  # Reset the form

elif page == "Stored Receipts":
    st.title("Stored Receipts")

    # Allow user to create a new folder
    new_folder_name = st.text_input("Enter folder name to create")
    if st.button("Create Folder"):
        if new_folder_name:
            # Save the folder name to the database
            folders_collection.insert_one({"folder_name": new_folder_name})
            st.success(f"Folder '{new_folder_name}' created successfully!")
        else:
            st.error("Folder name cannot be empty.")

    # Display existing folders and their receipts
    folders = list(folders_collection.find())
    for folder in folders:
        st.subheader(f"Folder: {folder['folder_name']}")

        # Fetch receipts for the current folder
        receipts = list(receipts_collection.find({"folder_name": folder['folder_name']}))
        for receipt in receipts:
            receipt["_id"] = str(receipt["_id"])
            with st.expander(f"Store: {receipt['store_name']} | Date & Time: {receipt['date_time']} | Total: {receipt['total']}"):
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

                # Display receipt image
                if "image_url" in receipt:
                    st.image(receipt["image_url"], caption="Receipt Image", width=300)

elif page == "Analyze":
    st.title("Analyze")

    # Create a form for the question input and submission
    with st.form(key="question_form"):
        question = st.text_input("Enter your question here")
        submit_button = st.form_submit_button(label="Ask")

    if submit_button and question:
        answer = analyze_data(question=question,
                              bamboollm_api_key=bamboollm_api_key)
        st.write(answer)

# Running the app - "streamlit run src/app.py"
