
def analyze_data(question, bamboollm_api_key, csv_file_path="csv_data/receipts_data.csv"):
    
    from pandasai import SmartDataframe
    from pandasai.llm import BambooLLM

    llm = BambooLLM(api_key=bamboollm_api_key)
    df = SmartDataframe(csv_file_path, config={"llm": llm})

    answer = df.chat(question)

    return answer