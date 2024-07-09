from utils import generate_text_from_image, config_model, extract_receipt_info
from get_receipt import detect_and_crop
import json
from dotenv import load_dotenv
import os
load_dotenv()
hf_token = os.getenv('HF_TOKEN')

# define the values
target_class_name = "book"

model, processor, device = config_model()

# Example usage
image_path = "image1"
image_path = f"images/{image_path}.jpg"  # Replace with your image path

# extract the portion of the image which contains the receipt
cropped_image_path = detect_and_crop(image_path=image_path,
                                     target_class_name=target_class_name)

extracted_text = generate_text_from_image(model=model, 
                                          image_path=cropped_image_path, 
                                          processor=processor, 
                                          device=device)

receipt_data = json.dumps(extracted_text)
receipt_data = json.loads(receipt_data)

final_data = extract_receipt_info(receipt_data)

# Save the final data to a result.json file
with open('result.json', 'w') as json_file:
    json.dump(final_data, json_file, indent=4)