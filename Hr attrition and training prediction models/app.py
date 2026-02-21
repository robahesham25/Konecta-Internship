import joblib
import pandas as pd
import numpy as np
import xgboost as xgb
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional
import os  
import google.generativeai as genai 
# from dotenv import load_dotenv

# load_dotenv()  

class HRFeatures(BaseModel):
    # From Employee.csv
    EmployeeID: str
    Age: int
    BusinessTravel: str
    Department: str
    DistanceFromHome_KM: int = Field(..., alias='DistanceFromHome (KM)')
    Education: int
    EducationField: str
    Gender: str
    JobRole: str
    MaritalStatus: str
    Salary: int
    StockOptionLevel: int
    OverTime: int
    YearsAtCompany: int
    YearsInMostRecentRole: int
    YearsSinceLastPromotion: int
    YearsWithCurrManager: int
    Ethnicity: str
    State: str
    
    # From PerformanceRating.csv
    EnvironmentSatisfaction: int
    JobSatisfaction: int
    RelationshipSatisfaction: int
    TrainingOpportunitiesWithinYear: int
    TrainingOpportunitiesTaken: int
    WorkLifeBalance: int
    SelfRating: int
    ManagerRating: int
    
    PerformanceID: Optional[str] = None
    ReviewDate: Optional[str] = None
    FirstName: Optional[str] = None
    LastName: Optional[str] = None
    HireDate: Optional[str] = None
    Attrition: Optional[int] = None

    
class AttritionRiskOutput(BaseModel):
    EmployeeID: str
    FullName: str  
    AttritionProbability: float


class TrainingNeedsOutput(BaseModel):
    EmployeeID: str
    FullName: str
    NeedsTrainingProbability: float

    class Config:
        allow_population_by_name = True

# --- 2. Initialize FastAPI App ---
app = FastAPI(
    title="HR AI API",
    description="Predicts Employee Attrition, Training Needs, and Analyzes Gaps."
)

# --- 3. Load ML Models & Artifacts (at startup) ---
try:
    attrition_model = joblib.load('xgboost_model_attrition.joblib')
    attrition_scaler = joblib.load('scaler_attrition.joblib')
    ATTRITION_FEATURES = joblib.load('attrition_feature_names.joblib')
    
    training_model = joblib.load('logistic_model_training.joblib')
    TRAINING_FEATURES = joblib.load('training_feature_names.joblib')
    
    print("ML models and artifacts loaded successfully.")
except FileNotFoundError as e:
    print(f"CRITICAL ERROR: Could not load ML models. {e}")
    attrition_model = None
    training_model = None

# --- 4. Load LLM Model & API Key (at startup) ---
llm_model = None

PROMPT_TEMPLATE = """
You are an expert HR Analyst. Your task is to analyze employee data to identify training gaps and recommend actions.

Based on the employee data provided below, perform the following 4-step reasoning:

**Employee Data:**
{employee_data_string}

---
**Reasoning and Analysis:**

**1. Identify the Gap:**
* First, state the primary performance concern. Is there a gap based on `ManagerRating <= 3` or `SelfRating <= 3`?
* Identify if this is a high attrition risk based on the `Attrition = 1` flag.

**2. Synthesize Context (The "Why"):**
* Analyze the relationship between performance (`ManagerRating`, `SelfRating`) and satisfaction (`JobSatisfaction`, `EnvironmentSatisfaction`). Are they aligned or conflicting?
* Examine the training history: `TrainingOpportunitiesWithinYear` vs. `TrainingOpportunitiesTaken`. Is there a missed opportunity?
* Look at workload and tenure: How might `OverTime`, `YearsAtCompany`, and `YearsSinceLastPromotion` be contributing to the performance gap?
* Consider the `JobRole`: What specific skills might be lacking for this role given the data?

**3. Identified Gaps (Summary List):**
* Based on your synthesis in Step 2, provide a clear, bulleted list of the specific skill or knowledge gaps this employee appears to have. (e.g., "Technical Product Knowledge," "Time Management," "Client Negotiation").

**4. Actionable Recommendations:**
* Based on the gaps identified in Step 3, recommend 2-3 specific, actionable steps for the employee's manager.
* Suggest specific training topics or interventions for each identified gap.

Format your output clearly with these four bolded headings.
"""

try:

    GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
    if not GOOGLE_API_KEY:
        print("WARNING: GOOGLE_API_KEY environment variable not set. /analyze_training_gaps will not work.")
    else:
        genai.configure(api_key=GOOGLE_API_KEY)
        llm_model = genai.GenerativeModel('gemini-2.5-flash')
        print("Generative AI model (gemini-pro) loaded successfully.")
except Exception as e:
    print(f"CRITICAL ERROR loading Generative AI model: {e}")


def preprocess_for_attrition(data_dict: dict) -> np.ndarray:
    df = pd.DataFrame([data_dict])
    categorical_cols = [
        'BusinessTravel', 'Department', 'EducationField', 'Gender', 
        'JobRole', 'MaritalStatus', 'Ethnicity', 'State'
    ]
    df_encoded = pd.get_dummies(df, columns=categorical_cols)
    df_aligned = df_encoded.reindex(columns=ATTRITION_FEATURES, fill_value=0)
    scaled_data = attrition_scaler.transform(df_aligned)
    return scaled_data

def preprocess_for_training(data_dict: dict) -> pd.DataFrame:
    df = pd.DataFrame([data_dict])
    df_aligned = df.reindex(columns=TRAINING_FEATURES, fill_value=0)
    return df_aligned


def load_and_prepare_data_for_bulk():
    """
    Reads the combined.csv file, restores categorical strings, 
    and prepares the list of dictionaries for bulk prediction.
    """
    DATA_CSV_PATH = "combined.csv" # Relative path to file in the same directory
    
    try:
        merged_df = pd.read_csv(DATA_CSV_PATH)
    except FileNotFoundError:
        raise HTTPException(status_code=500, detail=f"Data file '{DATA_CSV_PATH}' not found on server.")
  
    categorical_groups = {
        'BusinessTravel': [c for c in merged_df.columns if c.startswith('BusinessTravel_')],
        'Department': [c for c in merged_df.columns if c.startswith('Department_')],
        'EducationField': [c for c in merged_df.columns if c.startswith('EducationField_')],
        'Gender': [c for c in merged_df.columns if c.startswith('Gender_')],
        'JobRole': [c for c in merged_df.columns if c.startswith('JobRole_')],
        'MaritalStatus': [c for c in merged_df.columns if c.startswith('MaritalStatus_')],
        'Ethnicity': [c for c in merged_df.columns if c.startswith('Ethnicity_')],
        'State': [c for c in merged_df.columns if c.startswith('State_')]
    }
    
    for new_col, encoded_cols in categorical_groups.items():
        if any(col in merged_df.columns for col in encoded_cols):
            
            def find_category(row):
                for col in encoded_cols:
                    if col in row and (row[col] is True or row[col] == 1):
                        return col.split('_', 1)[1].strip()
                return "Unknown" 

            merged_df[new_col] = merged_df.apply(find_category, axis=1)
           
            merged_df = merged_df.drop(columns=encoded_cols, errors='ignore')
            
    
            if new_col == 'Gender':
                merged_df['Gender'] = merged_df['Gender'].replace('Unknown', 'Female') 


    if 'DistanceFromHome (KM)' in merged_df.columns:
         merged_df.rename(columns={'DistanceFromHome (KM)': 'DistanceFromHome_KM_ALIAS'}, inplace=True)

    cols_to_remove = ['PerformanceID', 'ReviewDate', 'HireDate', 'Attrition']
    df_ready = merged_df.drop(columns=cols_to_remove, errors='ignore')
 
    return df_ready


def find_employee_payload(employee_id: str, df: pd.DataFrame) -> dict:
    """
    Finds a specific employee's data row in the pre-loaded DataFrame and
    converts it into a JSON-ready dictionary (payload).
    """
  
    employee_row = df[df['EmployeeID'] == employee_id]
    
    if employee_row.empty:
        raise HTTPException(status_code=404, detail=f"Employee ID {employee_id} not found in dataset.")
       
    payload_dict = employee_row.iloc[0].to_dict()
   
    if 'DistanceFromHome_KM_ALIAS' in payload_dict:
        payload_dict['DistanceFromHome (KM)'] = payload_dict.pop('DistanceFromHome_KM_ALIAS')
    
    return payload_dict

# --- 6. API Endpoints ---

@app.get("/")
def home():
    return {"message": "HR AI API is running."}

@app.post("/predict_attrition")
def post_predict_attrition(employee: HRFeatures):
    """
    Predicts employee attrition risk.
    """
    if not attrition_model:
        raise HTTPException(status_code=500, detail="Attrition model not loaded.")

    try:
        data_dict = employee.dict(by_alias=True) 
        processed_data = preprocess_for_attrition(data_dict)
        dmatrix_data = xgb.DMatrix(processed_data, feature_names=ATTRITION_FEATURES)
        prediction_prob = attrition_model.predict(dmatrix_data)[0]
        attrition_risk = 1 if prediction_prob > 0.5 else 0
        
        return {
            'employee_id': data_dict.get('EmployeeID'),
            'attrition_probability': float(prediction_prob),
            'attrition_risk_prediction': int(attrition_risk)
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction error: {str(e)}")

@app.post("/predict_training")
def post_predict_training(employee: HRFeatures):
    """
    Predicts if an employee needs training.
    """
    if not training_model:
        raise HTTPException(status_code=500, detail="Training model not loaded.")

    try:
        data_dict = employee.dict(by_alias=True)
        processed_data = preprocess_for_training(data_dict)
        prediction = training_model.predict(processed_data)[0]
        prediction_prob = training_model.predict_proba(processed_data)[0][1]

        return {
            'employee_id': data_dict.get('EmployeeID'),
            'needs_training_prediction': int(prediction),
            'needs_training_probability': float(prediction_prob)
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction error: {str(e)}")


@app.post("/analyze_training_gaps")
async def post_analyze_training_gaps(employee: HRFeatures):
    """
    Analyzes training gaps using a Generative AI model.
    Requires GOOGLE_API_KEY to be set in the .env file.
    """
    if not llm_model:
        raise HTTPException(status_code=503, 
                            detail="Generative AI model is not configured or available. Check server logs.")

    try:
        data_dict = employee.dict(by_alias=True)
        employee_series = pd.Series(data_dict)
        data_string = employee_series.to_string()
        
        prompt = PROMPT_TEMPLATE.format(employee_data_string=data_string)
        response = await llm_model.generate_content_async(prompt)
       
        cleaned_analysis_text = response.text.replace('\\n', '\n')
      

        return {
            "employee_id": data_dict.get("EmployeeID"),
            "analysis": cleaned_analysis_text  
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating analysis: {str(e)}")
    

@app.post("/predict_attrition_bulk", response_model=List[AttritionRiskOutput])
def post_predict_attrition_bulk(employees: List[HRFeatures]):
    """
    Optimized to process the entire list of employees in one Pandas operation.
    """
    if not attrition_model:
        raise HTTPException(status_code=500, detail="Attrition model not loaded.")

    try:
        
        data_dicts = [emp.dict(by_alias=True) for emp in employees]
        df = pd.DataFrame(data_dicts) 
        cols_to_drop = ['Attrition', 'PerformanceID', 'ReviewDate', 'HireDate']
        df = df.drop(columns=cols_to_drop, errors='ignore')
        
        categorical_cols = [
            'BusinessTravel', 'Department', 'EducationField', 'Gender', 
            'JobRole', 'MaritalStatus', 'Ethnicity', 'State'
        ]
        
        df_encoded = pd.get_dummies(df.drop(columns=['EmployeeID', 'FirstName', 'LastName']), columns=categorical_cols)
        
     
        df_aligned = df_encoded.reindex(columns=ATTRITION_FEATURES, fill_value=0)
        processed_data_np = attrition_scaler.transform(df_aligned)

   
        dmatrix_data = xgb.DMatrix(processed_data_np, feature_names=ATTRITION_FEATURES)
        prediction_probs = attrition_model.predict(dmatrix_data)
    
        results = []
        for index, prob in enumerate(prediction_probs):
          
            emp_row = df.iloc[index]
            
            attrition_risk = 1 if prob > 0.5 else 0
         
            first_name = emp_row.get('FirstName', '')
            last_name = emp_row.get('LastName', '')
            full_name = f"{first_name} {last_name}".strip()
            if not full_name: full_name = "N/A"
            
            if attrition_risk == 1:
                results.append(AttritionRiskOutput(
                    EmployeeID=emp_row.get('EmployeeID'),
                    FullName=full_name,
                    AttritionProbability=float(prob),
                    IsHighRisk=int(attrition_risk)
                ))

        return results
    
    except Exception as e:
        print(f"Bulk Prediction Error: {e}")
        raise HTTPException(status_code=400, detail=f"Bulk prediction error: {str(e)}")



@app.get("/get_high_risk_list", response_model=List[AttritionRiskOutput])
def get_high_risk_list():
    """
    Reads the entire combined.csv file from the server, runs bulk prediction,
    and returns a filtered list of high-risk employees.
    """
    if not attrition_model:
        raise HTTPException(status_code=503, detail="Attrition model not loaded.")
        
    try:
        df = load_and_prepare_data_for_bulk()
        
        metadata_cols = ['EmployeeID', 'FirstName', 'LastName', 'DistanceFromHome_KM_ALIAS']
        df_features = df.drop(columns=metadata_cols, errors='ignore')
        
        df_aligned = df_features.reindex(columns=ATTRITION_FEATURES, fill_value=0)
        processed_data_np = attrition_scaler.transform(df_aligned)

        dmatrix_data = xgb.DMatrix(processed_data_np, feature_names=ATTRITION_FEATURES)
        prediction_probs = attrition_model.predict(dmatrix_data)
        
        results = []
        for index, prob in enumerate(prediction_probs):
            emp_row = df.iloc[index]
            attrition_risk = 1 if prob > 0.5 else 0
    
            first_name = emp_row.get('FirstName', '')
            last_name = emp_row.get('LastName', '')
            full_name = f"{first_name} {last_name}".strip()
            
            if attrition_risk == 1:
                results.append(AttritionRiskOutput(
                    EmployeeID=emp_row.get('EmployeeID'),
                    FullName=full_name if full_name else "N/A",
                    AttritionProbability=float(prob),
                    IsHighRisk=int(attrition_risk)
                ))

        return results
    
    except Exception as e:
        print(f"GET Bulk Prediction Error: {e}")
        raise HTTPException(status_code=500, detail=f"Server error during bulk prediction: {str(e)}")
    

@app.get("/get_training_needs_list", response_model=List[TrainingNeedsOutput])
def get_training_needs_list():
    """
    Reads the entire combined.csv file from the server, runs bulk prediction using the 
    Training Model, and returns a filtered list of employees who need training.
    """
    if not training_model:
        raise HTTPException(status_code=503, detail="Training model not loaded.")
        
    try:
        df = load_and_prepare_data_for_bulk()
        
        metadata_cols = ['EmployeeID', 'FirstName', 'LastName', 'DistanceFromHome_KM_ALIAS']
        
        df_features = df.drop(columns=metadata_cols, errors='ignore')
        
        df_aligned = df_features.reindex(columns=TRAINING_FEATURES, fill_value=0)
     
        prediction_probs = training_model.predict_proba(df_aligned)[:, 1]
        predictions = training_model.predict(df_aligned)
    
        results = []
        for index, prob in enumerate(prediction_probs):
            emp_row = df.iloc[index]
            prediction = predictions[index]
            
        
            first_name = emp_row.get('FirstName', '')
            last_name = emp_row.get('LastName', '')
            full_name = f"{first_name} {last_name}".strip()
            
            if prediction == 1:
                results.append(TrainingNeedsOutput(
                    EmployeeID=emp_row.get('EmployeeID'),
                    FullName=full_name if full_name else "N/A",
                    NeedsTrainingProbability=float(prob),
                    NeedsTrainingPrediction=int(prediction)
                ))

        return results
    
    except Exception as e:
        print(f"GET Training Prediction Error: {e}")
        raise HTTPException(status_code=500, detail=f"Server error during bulk training prediction: {str(e)}")
    
@app.get("/analyze_employee/{employee_id}")
async def get_analyze_employee_by_id(employee_id: str):
    """
    Retrieves all employee data internally based on the ID and runs the LLM analysis.
    Dashboard input: Employee ID only.
    """
    if not llm_model:
        raise HTTPException(status_code=503, detail="LLM not configured.")

    try:
        df_cleaned = load_and_prepare_data_for_bulk()
        
        employee_payload = find_employee_payload(employee_id, df_cleaned)

        employee_series = pd.Series(employee_payload)
        data_string = employee_series.to_string()
        
        prompt = PROMPT_TEMPLATE.format(employee_data_string=data_string)
        response = await llm_model.generate_content_async(prompt)
        
        cleaned_analysis_text = response.text.replace('\\n', '\n')
        
        return {
            "employee_id": employee_id,
            "analysis": cleaned_analysis_text
        }
    
    except HTTPException:
        raise 
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis error: {str(e)}")


@app.get("/predict_attrition/{employee_id}")
def get_predict_attrition_by_id(employee_id: str):
    """
    Retrieves all employee data internally and predicts attrition risk.
    Input: Employee ID via URL path.
    """
    if not attrition_model:
        raise HTTPException(status_code=503, detail="Attrition model not loaded.")

    try:
        df_cleaned = load_and_prepare_data_for_bulk()
      
        employee_payload = find_employee_payload(employee_id, df_cleaned)
        
        processed_data = preprocess_for_attrition(employee_payload) 
        
        dmatrix_data = xgb.DMatrix(processed_data, feature_names=ATTRITION_FEATURES)
        prediction_prob = attrition_model.predict(dmatrix_data)[0]
        attrition_risk = 1 if prediction_prob > 0.5 else 0
        
        return {
            'employee_id': employee_id,
            'attrition_probability': float(prediction_prob),
            'attrition_risk_prediction': int(attrition_risk)
        }
    except HTTPException:
        raise 
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")
    
@app.get("/predict_training/{employee_id}")
def get_predict_training_by_id(employee_id: str):
    """
    Retrieves all employee data internally and predicts training needs.
    Input: Employee ID via URL path.
    """
    if not training_model:
        raise HTTPException(status_code=503, detail="Training model not loaded.")

    try:
        df_cleaned = load_and_prepare_data_for_bulk()
     
        employee_payload = find_employee_payload(employee_id, df_cleaned)
        
        processed_data = preprocess_for_training(employee_payload)
        
        prediction = training_model.predict(processed_data)[0]
        prediction_prob = training_model.predict_proba(processed_data)[0][1]

        return {
            'employee_id': employee_id,
            'needs_training_prediction': int(prediction),
            'needs_training_probability': float(prediction_prob)
        }
    except HTTPException:
        raise 
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")