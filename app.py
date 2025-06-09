import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from src.diabetes import DataOutput_diabetes, prediction_diabetes, results_df_diabetes, roc_diabetes, shap_diabetes
from src.heart_failure import DataOutput_heart, prediction_heart, results_df_heart_failure, roc_heart, shap_heart
from src.stroke import DataOutput_stroke, prediction_stroke, results_df_stroke, roc_stroke, shap_stroke
#!pip install gdown
#import gdown
from PIL import Image

###################### model performance table ############################
def merge_results_dfs(disease_results):

    combined_df = pd.DataFrame()
    for disease, df in disease_results.items():
        df_copy = df.copy()
        df_copy["Disease"] = disease
        combined_df = pd.concat([combined_df, df_copy], ignore_index=True)
    return combined_df
def show_result(value):
    if value ==1:
        return "Positive"
    elif value ==0:
        return "Negative"
    else:
        return value
# Set up the page
st.set_page_config(page_title="HealthRadar360", layout="wide")
st.sidebar.title("🩺 HealthRadar360")

# Sidebar Navigation
page = st.sidebar.radio("Navigation", ["🏠 Home", "📝 Survey", "📊 Results", "⚙️ Models", "📬 Contact"])
X_train_diabetes, X_test_diabetes, Y_train_diabetes, Y_test_diabetes = DataOutput_diabetes()
X_train_heart, X_test_heart, Y_train_heart, Y_test_heart = DataOutput_heart()
X_train_stroke, X_test_stroke, Y_train_stroke, Y_test_stroke = DataOutput_stroke()

# 📸 3 images in columns 
# Image 1 #https://drive.google.com/file/d/1lupcOFRH-nP4gbWI2ebAsCxlvcLm6v_v/view?usp=sharing
#file_id1 = "1lupcOFRH-nP4gbWI2ebAsCxlvcLm6v_v"
#url1 = f"https://drive.google.com/uc?id={file_id1}"
#gdown.download(url1, "diabetes.png", quiet=False)

# Image 2 #https://drive.google.com/file/d/13uFV-ZIKXRPgyE8CYjJxKr1yzrWQlRp_/view?usp=sharing
#file_id2 = "13uFV-ZIKXRPgyE8CYjJxKr1yzrWQlRp_"
#url2 = f"https://drive.google.com/uc?id={file_id2}"
#gdown.download(url2, "heart.jpg", quiet=False)

# Image 3 #https://drive.google.com/file/d/1UO2Cc8kS5LT3YDEZ6Jdz2FtyiLJTC_-Z/view?usp=sharing
#file_id3 = "1UO2Cc8kS5LT3YDEZ6Jdz2FtyiLJTC_-Z"
#url3 = f"https://drive.google.com/uc?id={file_id3}"
#gdown.download(url3, "stroke.png", quiet=False)

# ----------------------------------------
# 🏠 HOME PAGE
# ----------------------------------------
if page == "🏠 Home":
  # Title
  st.title("🩺 HealthRadar360")

  # Introduction
  st.markdown("""
  ## Welcome to HealthRadar360

  HealthRadar360 is a data-driven tool designed to help you assess your risk for several major chronic diseases — including **stroke**, **diabetes**, and **heart disease**. Using advanced analytics and evidence-based models, we evaluate your health profile to give you personalized risk insights.
  """)



  # Load images
  image1 = Image.open("diabetes.png")
  image2 = Image.open("heart.jpg")
  image3 = Image.open("stroke.png")

  # Create 3 columns
  col1, col2, col3 = st.columns(3)

  with col1:
      st.image(image1, caption="Diabetes Prediction", use_container_width=True)

  with col2:
      st.image(image2, caption="Heart Failure Prediction", use_container_width=True)

  with col3:
      st.image(image3, caption="Stroke Prediction", use_container_width=True)






  st.markdown("""
  ---

  ### 🔍 How It Works

  To get your risk assessment:
  1. **Complete a brief health survey** covering your lifestyle, medical history, and family background.
  2. Based on your responses, HealthRadar360 will generate a personalized risk profile.
  3. You’ll receive clear, actionable insights to help guide your next steps.

  ---

  ### ⚠️ Disclaimer

  - This application is intended for **informational and educational purposes only** and should **not be used as a substitute for professional medical advice, diagnosis, or treatment**.
  - Always consult a qualified healthcare professional for medical concerns or before making health-related decisions.

  ---

  👈 **Click the survey tab to begin your assessment.**
  Your health journey starts now.
  """)

# ----------------------------------------
# 📝 SURVEY PAGE
# ----------------------------------------
elif page == "📝 Survey":
    st.title("📝 Health Survey")

    with st.form("health_survey_form"):
        st.subheader("General Info")
        age = st.number_input("Age", value = 30, min_value=1, max_value=120, step=1)
        gender = st.selectbox("Gender", ["Male", "Female"])

        st.subheader("Lifestyle")
        ever_married = st.radio(
                "Marriage History",
            ('Married/Was Married', 'Never Married'),
            help = 'Has the patient ever been married?'
        )
        Is_Urban_Residence = st.radio(
                "Location of Residence",
            ('Urban', 'Rural')
        )
        work_type = st.selectbox(
            label='Type of Work',
            options=('Self-Employed', 'Government Job','Private Industry','Never Worked')
            )

        st.subheader("Health Background")

        bmi = st.number_input("Your BMI", min_value=10.0, max_value=50.0,value =25.0,step=1.0)
        hypertension = st.radio("Hypertension", ["Yes", "No"])
        #diabetes = st.radio("Diabetes", ["Yes", "No"])
        heart = st.radio("Heart disease", ["Yes", "No"])
        #stroke = st.radio("Stroke", ["Yes", "No"])

        smoking_status = st.radio("Do you smoke?", ['Never Smoked', 'Ex-Smoker','Smoker','Unknown'])
        #alcohol = st.radio("Do you drink alcohol?", ["Yes", "No"])
        #exercise = st.selectbox("Exercise frequency per week", ["None", "1-2 times", "3-4 times", "5+ times"])

        st.subheader("General Clinical Metrics")

        HbA1c_level = st.slider(label = "HbA1c Level",min_value = 3.5,
            max_value = 9.0,
            value=6.0,
            step=0.1)
        blood_glucose_level = st.slider(
            label = "Glucose Level",
            min_value = 50.0,
            max_value = 280.0,
            value=130.0,
            step=0.1
        )

        st.subheader("Heart Disease Specific")

        chest_pain_type = st.radio("ChestPainType", ["TA: Typical Angina", "ATA: Atypical Angina", "NAP: Non-Anginal Pain", "ASY: Asymptomatic"])
        resting_bp = st.number_input("RestingBP (Resting Blood Pressure in mm Hg)", min_value=0, value=100)
        cholesterol = st.number_input("Cholesterol (Serum cholesterol in mg/dl)", min_value=0, value=140)
        fasting_bs = st.radio("FastingBS (Is fasting blood sugar > 120 mg/dl?)", ["Yes", "No"])
        resting_ecg = st.radio("RestingECG", ["Normal", "ST: ST-T wave abnormality", "LVH: Left Ventricular Hypertrophy"])
        max_hr = st.number_input("MaxHR (Maximum Heart Rate Achieved)", min_value=60, max_value=300, value=150,)
        exercise_angina = st.radio("ExerciseAngina (Exercise-induced angina)", ["Yes", "No"])
        oldpeak = st.number_input("Oldpeak (ST depression induced by exercise)", format="%.2f", value=0.8)
        st_slope = st.radio("ST_Slope (Slope of the peak exercise ST segment)", ["Up", "Flat", "Down"])

        submit = st.form_submit_button("Submit Survey")

    if submit==True:
      with st.spinner('Analysing ...'):

        if gender == 'Male':
            gender_Female = 0
            gender_Male = 1
            gender_Other = 0
        elif gender == 'Female':
            gender_Female = 1
            gender_Male = 0
            gender_Other = 0
        else:
            gender_Female = 0
            gender_Male = 0
            gender_Other = 1

        if hypertension == 'Yes':
            hypertension = 1
        else:
            hypertension = 0

        if heart == 'Yes':
            heart_disease = 1
        else:
            heart_disease = 0

        if ever_married == 'Married/Was Married':
            ever_married_Yes = 1
            ever_married_No = 0
        else:
            ever_married_Yes = 0
            ever_married_No = 1

        if Is_Urban_Residence == 'Urban':
            Residence_type_Urban = 1
            Residence_type_Rural = 0
        else:
            Residence_type_Urban = 0
            Residence_type_Rural = 1

        if work_type == 'Self-Employed':
            work_type_Government_Job=0
            work_type_Never_worked=0
            work_type_Private_Industry=0
            work_type_Self_employed=1
        elif work_type == 'Government Job':
            work_type_Government_Job=1
            work_type_Never_worked=0
            work_type_Private_Industry=0
            work_type_Self_employed=0
        elif work_type == 'Private Industry':
            work_type_Government_Job=0
            work_type_Never_worked=0
            work_type_Private_Industry=1
            work_type_Self_employed=0
        else:
            work_type_Government_Job=0
            work_type_Never_worked=1
            work_type_Private_Industry=0
            work_type_Self_employed=0

        if fasting_bs == 'Yes':
            fasting_bs = 1
        else:
            fasting_bs = 0

        [smoking_status_Former_Smoker, smoking_status_Never_Smoked,smoking_status_Smoker,smoking_status_Unknown] = [1 if smoking_status in item else 0 for item in ['Ex-Smoker', 'Never Smoked', 'Smoker', 'Unknown']]
        [chest_pain_type_ta,chest_pain_type_ata,chest_pain_type_nap,chest_pain_type_asy] = [1 if chest_pain_type in item else 0 for item in ["TA: Typical Angina", "ATA: Atypical Angina", "NAP: Non-Anginal Pain", "ASY: Asymptomatic"]]
        [resting_ecg_normal,resting_ecg_st,resting_ecg_lvh] = [1 if resting_ecg in item else 0 for item in ["Normal", "ST: ST-T wave abnormality", "LVH: Left Ventricular Hypertrophy"]]
        [exercise_angina_y,exercise_angina_n] = [1 if exercise_angina in item else 0 for item in ["Yes", "No"]]
        [st_slope_up,st_slope_flat,st_slope_down] = [1 if st_slope in item else 0 for item in ["Up", "Flat", "Down"]]

        st.session_state["survey_completed"] = True

        # Diabetes
        x_diabetes = [age,
        hypertension,
        heart_disease,
        bmi,
        HbA1c_level,
        blood_glucose_level,
        gender_Female,
        gender_Male,
        gender_Other,
        smoking_status_Unknown,
        smoking_status_Smoker,
        smoking_status_Former_Smoker|smoking_status_Smoker,
        smoking_status_Former_Smoker,
        smoking_status_Never_Smoked,
        smoking_status_Former_Smoker|smoking_status_Never_Smoked]

        dfx_diabetes = pd.DataFrame(data = [x_diabetes],columns = ['age', 'hypertension', 'heart_disease', 'bmi', 'HbA1c_level',
       'blood_glucose_level', 'gender_Female', 'gender_Male', 'gender_Other',
       'smoking_history_No Info', 'smoking_history_current',
       'smoking_history_ever', 'smoking_history_former',
       'smoking_history_never', 'smoking_history_not current'])
        Y_RF_diabetes,Y_XG_diabetes,Y_LR_diabetes = prediction_diabetes(dfx_diabetes)
        st.session_state["Y_RF_diabetes"] = Y_RF_diabetes
        st.session_state["Y_XG_diabetes"] = Y_XG_diabetes
        st.session_state["Y_LR_diabetes"] = Y_LR_diabetes

        # Heart Failure
        x_heart = [age,
        resting_bp,
        cholesterol,
        fasting_bs,
        max_hr,
        oldpeak,
        gender_Female,
        gender_Male,
        chest_pain_type_asy,
        chest_pain_type_ata,
        chest_pain_type_nap,
        chest_pain_type_ta,
        resting_ecg_lvh,
        resting_ecg_normal,
        resting_ecg_st,
        exercise_angina_n,
        exercise_angina_y,
        st_slope_down,
        st_slope_flat,
        st_slope_up]
        dfx_heart = pd.DataFrame(data = [x_heart],columns =
        ['Age', 'RestingBP', 'Cholesterol', 'FastingBS', 'MaxHR', 'Oldpeak',
       'Sex_F', 'Sex_M', 'ChestPainType_ASY', 'ChestPainType_ATA',
       'ChestPainType_NAP', 'ChestPainType_TA', 'RestingECG_LVH',
       'RestingECG_Normal', 'RestingECG_ST', 'ExerciseAngina_N',
       'ExerciseAngina_Y', 'ST_Slope_Down', 'ST_Slope_Flat', 'ST_Slope_Up'])
        Y_RF_heart,Y_XG_heart,Y_LR_heart = prediction_heart(dfx_heart)
        st.session_state["Y_RF_heart"] = Y_RF_heart
        st.session_state["Y_XG_heart"] = Y_XG_heart
        st.session_state["Y_LR_heart"] = Y_LR_heart

        # Stroke
        x_stroke = [age,
        hypertension,
        heart_disease,
        blood_glucose_level,
        bmi,
        gender_Female,
        gender_Male,
        gender_Other,
        ever_married_No,
        ever_married_Yes,
        work_type_Government_Job,
        work_type_Never_worked,
        work_type_Private_Industry,
        work_type_Self_employed,
        0,
        Residence_type_Rural,
        Residence_type_Urban,
        smoking_status_Unknown,
        smoking_status_Former_Smoker,
        smoking_status_Never_Smoked,
        smoking_status_Smoker,]

        dfx_stroke = pd.DataFrame(data = [x_stroke],columns =
        ['age', 'hypertension', 'heart_disease', 'avg_glucose_level', 'bmi',
       'gender_Female', 'gender_Male', 'gender_Other', 'ever_married_No',
       'ever_married_Yes', 'work_type_Govt_job', 'work_type_Never_worked',
       'work_type_Private', 'work_type_Self-employed', 'work_type_children',
       'Residence_type_Rural', 'Residence_type_Urban',
       'smoking_status_Unknown', 'smoking_status_formerly smoked',
       'smoking_status_never smoked', 'smoking_status_smokes'])
        Y_RF_stroke,Y_XG_stroke,Y_LR_stroke = prediction_stroke(dfx_stroke)
        st.session_state["Y_RF_stroke"] = Y_RF_stroke
        st.session_state["Y_XG_stroke"] = Y_XG_stroke
        st.session_state["Y_LR_stroke"] = Y_LR_stroke

        # New patient data
        st.session_state["new_patient"] = {
            'bmi': bmi,
            'HbA1c_level': HbA1c_level,
            'blood_glucose_level': blood_glucose_level,
            'RestingBP': resting_bp,
            'Cholesterol': cholesterol,
            'MaxHR': max_hr,
            'Oldpeak': oldpeak
        }

    if submit:
        st.session_state["survey_completed"] = True
        st.success("Survey submitted! Go to the Results page.")

# ----------------------------------------
# 📊 RESULTS PAGE
# ----------------------------------------
elif page == "📊 Resultsold":
    st.title("📊 Risk Assessment Results")

    if st.session_state.get("survey_completed", True):
        Y_RF_diabetes = st.session_state.get("Y_RF_diabetes", "Not calculated")
        Y_XG_diabetes = st.session_state.get("Y_XG_diabetes", "Not calculated")
        Y_LR_diabetes = st.session_state.get("Y_LR_diabetes", "Not calculated")
        Y_RF_heart = st.session_state.get("Y_RF_heart", "Not calculated")
        Y_XG_heart = st.session_state.get("Y_XG_heart", "Not calculated")
        Y_LR_heart = st.session_state.get("Y_LR_heart", "Not calculated")
        Y_RF_stroke = st.session_state.get("Y_RF_stroke", "Not calculated")
        Y_XG_stroke = st.session_state.get("Y_XG_stroke", "Not calculated")
        Y_LR_stroke = st.session_state.get("Y_LR_stroke", "Not calculated")

        st.markdown("""
        Based on your responses, here is a **sample risk overview**:
        """)

        #fig_shap_diabetes = plot_shap_diabetes()
        dcol1, dcol2 = st.columns([1, 1])
        with dcol1:

              st.subheader("🩸 **Diabetes Risk**:")
              st.write(f"Random Forest Prediction: **{Y_RF_diabetes}**")
              st.write(f"XGBoost Prediction: **{Y_XG_diabetes}**")
              st.write(f"Logistic Regression Prediction: **{Y_LR_diabetes}**")
              st.write(f"")

              st.subheader("❤️ **Heart Disease Risk**:")
              st.write(f"Random Forest Prediction: **{Y_RF_heart}**")
              st.write(f"XGBoost Prediction: **{Y_XG_heart}**")
              st.write(f"Logistic Regression Prediction: **{Y_LR_heart}**")
              st.write(f"")

              st.subheader("🧠 **Stroke Risk**:")
              st.write(f"Random Forest Prediction: **{Y_RF_heart}**")
              st.write(f"XGBoost Prediction: **{Y_XG_heart}**")
              st.write(f"Logistic Regression Prediction: **{Y_LR_heart}**")
              st.write(f"")
        with dcol2:
              #boxplot
              st.subheader("📊 Patient Values vs Healthy Population")
              numerical = ['bmi', 'HbA1c_level', 'blood_glucose_level', 'RestingBP', 'Cholesterol', 'MaxHR', 'Oldpeak']
              #new_patient
              new_patient = st.session_state.get("new_patient", "Not calculated")
              # Summary statistics ## consider with healthy range
              summary_stats = {
                  'bmi': {'min': 18.5, 'Q1': 20.62, 'median': 22.21, 'Q3': 23.58, 'max': 29.9},
                  'HbA1c_level': {'min': 3.5, 'Q1': 4.0, 'median': 4.8, 'Q3': 5.0, 'max': 6.5},
                  'blood_glucose_level': {'min': 70, 'Q1': 85.0, 'median': 100.0, 'Q3': 130.0, 'max': 140},
                  'RestingBP': {'min': 94, 'Q1': 110.0, 'median': 120.0, 'Q3': 120.0, 'max': 130},
                  'Cholesterol': {'min': 126, 'Q1': 167.0, 'median': 182.0, 'Q3': 195.0, 'max': 200},
                  'MaxHR': {'min': 100, 'Q1': 135.0, 'median': 150.0, 'Q3': 163.0, 'max': 180},
                  'Oldpeak': {'min': 0.0, 'Q1': 0.0, 'median': 0.0, 'Q3': 0.2, 'max': 1.0}
              }
              # Set theme
              sns.set(style="white")
              # Create subplots
              fig, axes = plt.subplots(len(numerical), 1, figsize=(10, 8))
              palette = "#4C72B0"
              for i, feature in enumerate(numerical):
                  ax = axes[i]
                  # Prepare box data in bxp format
                  stats = summary_stats[feature]
                  box_data = [{
                      'label': feature,
                      'whislo': stats['min'],
                      'q1': stats['Q1'],
                      'med': stats['median'],
                      'q3': stats['Q3'],
                      'whishi': stats['max'],
                      'fliers': []
                  }]
                  # Draw boxplot with nicer styling
                  ax.bxp(
                      box_data,
                      vert=False,
                      showfliers=False,
                      patch_artist=True,
                      boxprops=dict(facecolor=palette, alpha=0.9, linewidth=1, edgecolor='black'),
                      whiskerprops=dict(linewidth=1),
                      capprops=dict(linewidth=1),
                      medianprops=dict(color='black', linewidth=1),
                      widths=0.6
                  )
                  # Add patient value
                  ax.scatter(new_patient[feature], 1, color='red', s=60, label='Patient Value' if i == 0 else "", zorder=5)
                  # Set axis limits for better view
                  if feature == 'bmi':
                      ax.set_xlim(10, 40)
                  elif feature == 'HbA1c_level':
                      ax.set_xlim(3, 10)
                  elif feature == 'blood_glucose_level':
                      ax.set_xlim(50, 200)
                  elif feature == 'RestingBP':
                      ax.set_xlim(80, 160)
                  elif feature == 'Cholesterol':
                      ax.set_xlim(120, 260)
                  elif feature == 'MaxHR':
                      ax.set_xlim(80, 200)
                  elif feature == 'Oldpeak':
                      ax.set_xlim(0, 3)
                  # Title and axis
                  ax.set_title(feature, fontsize=12, loc='left')
                  ax.set_xlabel('')
                  ax.set_yticks([])
                  sns.despine(left=True)
              # Add legend once
              handles, labels = axes[0].get_legend_handles_labels()
              if handles:
                  fig.legend(handles, labels, loc='upper right', bbox_to_anchor=(0.98, 0.93), fontsize=9, frameon=False)
              # Title and spacing
              #fig.suptitle('Patient Value vs Healthy Population', fontsize=14)
              fig.tight_layout(rect=[0, 0, 0.95, 0.90])
              st.pyplot(fig)
        col1, col2 = st.columns([1, 1])
        #st.pyplot(shap_diabetes)
        #st.pyplot(shap_heart)
        #st.pyplot(shap_stroke)
        #with col1:
            #st.pyplot(shap_diabetes)

        #with col2:
            #st.pyplot(shap_heart)

        #with col1:
            #st.pyplot(shap_stroke)

    else:
        st.warning("🚨 Please complete the survey first to view your results.")
elif page == "📊 Results":
    st.title("📊 Risk Assessment Results")

    if st.session_state.get("survey_completed", True):
        Y_RF_diabetes = st.session_state.get("Y_RF_diabetes", "Not calculated")
        Y_XG_diabetes = st.session_state.get("Y_XG_diabetes", "Not calculated")
        Y_LR_diabetes = st.session_state.get("Y_LR_diabetes", "Not calculated")
        Y_RF_heart = st.session_state.get("Y_RF_heart", "Not calculated")
        Y_XG_heart = st.session_state.get("Y_XG_heart", "Not calculated")
        Y_LR_heart = st.session_state.get("Y_LR_heart", "Not calculated")
        Y_RF_stroke = st.session_state.get("Y_RF_stroke", "Not calculated")
        Y_XG_stroke = st.session_state.get("Y_XG_stroke", "Not calculated")
        Y_LR_stroke = st.session_state.get("Y_LR_stroke", "Not calculated")

        st.markdown("""
        Based on your responses, here is a **sample risk overview**:
        """)

        #fig_shap_diabetes = plot_shap_diabetes()
        dcol1, dcol2, dcol3 = st.columns([1, 1, 1])
        with dcol1:
              st.subheader("🩸 **Diabetes Risk**:")
              st.write(f"Random Forest Prediction: **{show_result(Y_RF_diabetes)}**")
              st.write(f"XGBoost Prediction: **{show_result(Y_XG_diabetes)}**")
              st.write(f"Logistic Regression Prediction: **{show_result(Y_LR_diabetes)}**")
              st.write(f"")
        with dcol2:
              st.subheader("❤️ **Heart Disease Risk**:")
              st.write(f"Random Forest Prediction: **{show_result(Y_RF_heart)}**")
              st.write(f"XGBoost Prediction: **{show_result(Y_XG_heart)}**")
              st.write(f"Logistic Regression Prediction: **{show_result(Y_LR_heart)}**")
              st.write(f"")
        with dcol3:
              st.subheader("🧠 **Stroke Risk**:")
              st.write(f"Random Forest Prediction: **{show_result(Y_RF_heart)}**")
              st.write(f"XGBoost Prediction: **{show_result(Y_XG_heart)}**")
              st.write(f"Logistic Regression Prediction: **{show_result(Y_LR_heart)}**")
              st.write(f"")

        dcol1, dcol2,dcol3 = st.columns([1, 0.2,1])
        with dcol1:
              st.pyplot(shap_diabetes)

              st.pyplot(shap_heart)

              st.pyplot(shap_stroke)
        with dcol2:
              st.write("    ")
        with dcol3:
              #boxplot
              st.subheader("📊 Patient Values vs Healthy Population")
              numerical = ['bmi', 'HbA1c_level', 'blood_glucose_level', 'RestingBP', 'Cholesterol', 'MaxHR', 'Oldpeak']
              #new_patient
              new_patient = st.session_state.get("new_patient", "Not calculated")
              # Summary statistics ## consider with healthy range
              summary_stats = {
                  'bmi': {'min': 18.5, 'Q1': 20.62, 'median': 22.21, 'Q3': 23.58, 'max': 29.9},
                  'HbA1c_level': {'min': 3.5, 'Q1': 4.0, 'median': 4.8, 'Q3': 5.0, 'max': 6.5},
                  'blood_glucose_level': {'min': 70, 'Q1': 85.0, 'median': 100.0, 'Q3': 130.0, 'max': 140},
                  'RestingBP': {'min': 94, 'Q1': 110.0, 'median': 120.0, 'Q3': 120.0, 'max': 130},
                  'Cholesterol': {'min': 126, 'Q1': 167.0, 'median': 182.0, 'Q3': 195.0, 'max': 200},
                  'MaxHR': {'min': 100, 'Q1': 135.0, 'median': 150.0, 'Q3': 163.0, 'max': 180},
                  'Oldpeak': {'min': 0.0, 'Q1': 0.0, 'median': 0.0, 'Q3': 0.2, 'max': 1.0}
              }
              # Set theme
              sns.set(style="white")
              # Create subplots
              fig, axes = plt.subplots(len(numerical), 1, figsize=(9, 12))
              palette = "#4C72B0"
              for i, feature in enumerate(numerical):
                  ax = axes[i]
                  # Prepare box data in bxp format
                  stats = summary_stats[feature]
                  box_data = [{
                      'label': feature,
                      'whislo': stats['min'],
                      'q1': stats['Q1'],
                      'med': stats['median'],
                      'q3': stats['Q3'],
                      'whishi': stats['max'],
                      'fliers': []
                  }]
                  # Draw boxplot with nicer styling
                  ax.bxp(
                      box_data,
                      vert=False,
                      showfliers=False,
                      patch_artist=True,
                      boxprops=dict(facecolor=palette, alpha=0.9, linewidth=1, edgecolor='black'),
                      whiskerprops=dict(linewidth=1),
                      capprops=dict(linewidth=1),
                      medianprops=dict(color='black', linewidth=1),
                      widths=0.6
                  )
                  # Add patient value
                  ax.scatter(new_patient[feature], 1, color='red', s=60, label='Patient Value' if i == 0 else "", zorder=5)
                  # Set axis limits for better view
                  if feature == 'bmi':
                      ax.set_xlim(10, 40)
                  elif feature == 'HbA1c_level':
                      ax.set_xlim(3, 10)
                  elif feature == 'blood_glucose_level':
                      ax.set_xlim(50, 200)
                  elif feature == 'RestingBP':
                      ax.set_xlim(80, 160)
                  elif feature == 'Cholesterol':
                      ax.set_xlim(120, 260)
                  elif feature == 'MaxHR':
                      ax.set_xlim(80, 200)
                  elif feature == 'Oldpeak':
                      ax.set_xlim(0, 3)
                  # Title and axis
                  ax.set_title(feature, fontsize=12, loc='left')
                  ax.set_xlabel('')
                  ax.set_yticks([])
                  sns.despine(left=True)
              # Add legend once
              handles, labels = axes[0].get_legend_handles_labels()
              if handles:
                  fig.legend(handles, labels, loc='upper right', bbox_to_anchor=(0.98, 0.93), fontsize=9, frameon=False)
              # Title and spacing
              #fig.suptitle('Patient Value vs Healthy Population', fontsize=14)
              fig.tight_layout(rect=[0, 0, 0.95, 0.90])
              st.pyplot(fig)

    else:
        st.warning("🚨 Please complete the survey first to view your results.")

# ----------------------------------------
# ⚙️ MODELS PAGE
# ----------------------------------------
elif page == "⚙️ Models":
        st.title("⚙️ Machine Learning Models")
        # Combine performance results
        combined_results_df = merge_results_dfs({
            "Diabetes": results_df_diabetes,
            "Heart Failure": results_df_heart_failure,
            "Stroke": results_df_stroke
        })
        # Move 'Disease' to the first position
        cols = combined_results_df.columns.tolist()
        cols.insert(0, cols.pop(cols.index('Disease')))
        combined_results_df = combined_results_df[cols]
        # Display in Streamlit
        st.subheader("📈 Model Performance Summary")
        st.dataframe(combined_results_df, use_container_width=True)

        # Figure
        #fig = plot_roc_curve()
        st.subheader("📈 ROC Curve Comparison Across Models")

        col1, col2, col3 = st.columns([1, 1, 1])
        with col1:
            st.pyplot(roc_diabetes)

        with col2:
            st.pyplot(roc_heart)

        with col3:
            st.pyplot(roc_stroke)
elif page == "📬 Contact":
    st.markdown("## 📬 Contact Me")
    st.write("Have a question or want to collaborate? Send me a message!")
    contact_form = """
    <style>
        .contact-form {
            max-width: 600px;
            margin: 0 auto;
        }
        .contact-form input, .contact-form textarea {
            width: 100%;
            padding: 0.75rem;
            margin-bottom: 1rem;
            border: 1px solid #ccc;
            border-radius: 8px;
        }
        .contact-form button {
            padding: 0.75rem;
            width: 100%;
            background-color: #333;
            color: white;
            border: none;
            border-radius: 8px;
            font-size: 1rem;
            cursor: pointer;
        }
        .contact-form button:hover {
            background-color: #555;
        }
    </style>

    <form class="contact-form" target="_blank" action="https://formsubmit.co/healthradar360@hotmail.com" method="POST">
        <input type="text" name="name" placeholder="Full Name" required>
        <input type="email" name="email" placeholder="Email Address" required>
        <textarea name="message" rows="8" placeholder="Your Message" required></textarea>
        <button type="submit">📩 Submit</button>
    </form>
    """
    st.markdown(contact_form, unsafe_allow_html=True)


st.markdown("---\n*Powered by HealthRadar360 · Built with Streamlit*", unsafe_allow_html=True)


