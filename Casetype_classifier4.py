"""
@author: Karan Kadam

"""


import os
import logging
from logging.handlers import TimedRotatingFileHandler
from langchain_openai import AzureChatOpenAI
import json
import yaml
import openai
#from langchain_openai import AzureChatOpenAI
from firm_case_classifier_api_v8 import process_query
from azure.identity import AzureCliCredential
from azure.keyvault.secrets import SecretClient


# Configure logging with a TimedRotatingFileHandler
logging.basicConfig(
    level=logging.INFO,
    handlers=[
        TimedRotatingFileHandler(
            filename='app.log',
            when='W0',  # Rotate logs on a weekly basis, starting on Monday
            backupCount=1  # Retain one backup log file (the current week's log)
        )
    ],
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class caseClassifier:
    def __init__(self):
        # Key Vault URL
        key_vault_url = "https://caseratekeyvault.vault.azure.net/"
        credential = AzureCliCredential()
        # Create a SecretClient using the Key Vault URL and credential
        client = SecretClient(vault_url=key_vault_url, credential=credential)

        self.db_path = client.get_secret("pl-db-path").value
        self.OPENAI_API_TYPE = client.get_secret("pl-azure-api-type").value
        self.OPENAI_API_KEY = client.get_secret("pl-open-api-key").value
        #self.OPENAI_DEPLOYMENT_VERSION = client.get_secret("pl-openai-deployment-version").value
        self.OPENAI_DEPLOYMENT_ENDPOINT = client.get_secret("pl-openai-deployment-endpoint").value
        #self.OPENAI_DEPLOYMENT_NAME = client.get_secret("pl-openai-deployment-name").value
        #self.OPENAI_MODEL_NAME = client.get_secret("pl-openai-model").value
        self.OPENAI_DEPLOYMENT_VERSION = client.get_secret("pl-openai-deployment-version-4o-mini").value
        #self.OPENAI_DEPLOYMENT_ENDPOINT = client.get_secret("pl-openai-deployment-endpoint").value
        self.OPENAI_DEPLOYMENT_NAME = client.get_secret("deployment-name-4o-mini").value
        self.OPENAI_MODEL_NAME = client.get_secret("modelname-4o-mini").value
        self.OPENAI_ADA_EMBEDDING_DEPLOYMENT_NAME = client.get_secret("pl-openai-ada-enbedding-deployment").value
        self.OPENAI_ADA_EMBEDDING_MODEL_NAME = client.get_secret("pl-openai-ada-embedding-model-name").value


        os.environ["OPENAI_API_KEY"] = self.OPENAI_API_KEY
        os.environ["OPENAI_DEPLOYMENT_ENDPOINT"] = self.OPENAI_DEPLOYMENT_ENDPOINT
        os.environ["OPENAI_DEPLOYMENT_NAME"] = self.OPENAI_DEPLOYMENT_NAME
        os.environ["OPENAI_MODEL_NAME"] = self.OPENAI_MODEL_NAME
        os.environ["OPENAI_DEPLOYMENT_VERSION"] = self.OPENAI_DEPLOYMENT_VERSION

        # self.custom_prompt_template = """
        # Given the following legal case description: "{question}"  
        # Classify this legal case into one of the following Case types based on the presence of relevant keywords:
        
        # 1. **Workers Compensation**:  
        # - Keywords: injured at work, workplace injury, workers' comp, on-the-job, occupational injury, compensation, work-related injury
        
        # 2. **Medical Malpractice**:  
        # - Keywords: misdiagnosis, malpractice, wrong treatment, surgical error, medication error, hospital, doctor, nurse, improper care, birth injury, delayed diagnosis, incorrect prescription
        
        # 3. **Employment Law**:  
        # - Keywords: discrimination, harassment, wrongful termination, fired, laid off, hostile work environment, retaliation, sexual harassment, unequal pay, whistleblower, race, gender, disability
        
        # 4. **Product Liability**:  
        # - Keywords: defective product, malfunction, injury from product, unsafe, recall, failure to warn, design defect, manufacturing defect, consumer safety
        
        # 5. **Personal Injury**:  
        # - Keywords: accident, injury, hurt, fall, slip, trip, burn, broken bone, concussion, negligence, pain, suffering, medical bills, lost wages, whiplash, liability, car accident, car wreck, car crash, motor vehicle accident, motorcycle, ice, dog, pitbull, rotweiler, attack, machine, malfunction, forklift
        
        # 6. **Mass Torts**:  
        # - Keywords: talc, baby powder, mesothelioma, ovarian cancer, hip replacement, knee replacement, cancer, leukemia, b-cell, t-cell, roundup, pesticide, agriculture, non-hodgkins, monsanto, mesh, transvaginal, bladder sling, zantac, valsartan, ranitidine, paraquat, parkinsons, lewy, CPAP, ABAP, BiPAP, ventilator, kidney disease, lung disease, reactive airway, acute respiratory distress, multiple myeloma, elmiron, macular, lejeune, longshoreman, hair relaxer, SJS, Stevens-Johnson Syndrome, Toxic Epidermal Necrolysis, Heparin, silicosis, Hypoxic-ischemic encephalopathy
        
        # 7. **Nursing Home**:  
        # - Keywords: neglect, bed sore, pressure sore, abuse, mistreatment, pressure ulcer, malnutrition, overmedication, unexplained injuries, understaffed
        
        # Analyze the legal case description, search for relevant keywords, and classify it into the most appropriate potential claim type.  
        # If no keywords match, respond with "No matching claim type found."
        
        # The output should be in the following JSON format without the json keyword and the JSON structure must have the following key values:
        # {
        #     "CaseType" : "Case Type here",
        #     "Explanation" : "Explain your answer here with detailed reasoning behind the case and why it falls under this category."
        # }

        # """
        self.custom_prompt_template = """
Given the following legal case description: "{question}"
Your task is to analyze the provided description and indicate whether it resembles a legal case.
Consider elements such as parties involved, legal issues, relevant laws court proceedings, or any other factors that
typically define a legal case. The goal is to screen descriptions from existing Social Security Disability intakes, 
to see if they imply the client may also be a candidate for Workers' Compensation, Long-Term Disability through 
a private policy with an insurance company, or another type of legal matter or injury claim that can be referred 
out to another law firm. If the description aligns with what you understand as a legal case, predict Case Type,
Case Rating for the description as per given guidelines below.

Case Type is depends on relevant keywords in description. classify description into one of the following Case types based on the presence of relevant keywords:

The Case Type is "Workers Compensation" when employees seek benefits for job-related injuries or illnesses. This includes cases where the employee is claiming medical expenses, lost wages, or 
rehabilitation due to work-related issues, regardless of employer fault.
    - Automatic Classification Triggers:
        - If the term "SST" appears in the description, it indicates an existing Workers Compensation client. Therefore, set the Case Type to "Workers Compensation" automatically.
        - If the description includes "We referred this client out for a Workers Compensation claim," Workers Compensation need not be reclassified, as this has already been addressed.

    - Keywords to Confirm Classification:
        - To validate "Workers Compensation" as the case type, look for keywords in the description that indicate a work-related injury or illness:
            - "Injured at work"
            - "workplace injury"
            - "workers comp"
            - "on-the-job"
            - "occupational injury"
            - "compensation"
            - "work-related injury"

    - Avoiding Misclassification: 
        - If no Workers Compensation keywords are present, and the context does not directly indicate a work-related injury, avoid assuming other case types (e.g., "Personal Injury") without further clarification.

Case Type is "Medical Malpractice", When a healthcare provider’s negligence deviates from the standard of care, causing harm. Proving duty, breach, causation, and damages is essential.
   - Keywords look for are:
    - misdiagnosis
    - malpractice
    - wrong treatment
    - surgical error
    - medication error
    - hospital
    - doctor
    - nurse
    - improper care
    - birth injury
    - delayed diagnosis
    - incorrect prescription

Case Type is "Employment Law", When legal issues arise between employers and employees, covering matters like wrongful termination, discrimination, harassment, wages, and workplace rights and protections
   - Keywords look for are:
    - discrimination
    - harassment
    - wrongful termination
    - fired
    - laid off
    - hostile work environment
    - retaliation
    - sexual harassment
    - unequal pay
    - whistleblower
    - race
    - gender
    - disability

Case Type is "Product Liability", When manufacturers or sellers are held accountable for injuries caused by defective or unsafe products, involving design flaws, manufacturing defects, or inadequate warnings.
   - Keywords look for are:
    - defective product
    - malfunction
    - injury from product
    - unsafe, recall
    - failure to warn
    - design defect
    - manufacturing defect
    - consumer safety

Case Type is "Personal Injury", When  individuals seek compensation for injuries caused by another’s negligence, accident, covering medical expenses, lost wages, pain and suffering, and related damages.
   - Keywords look for are:
    - accident
    - injury
    - hurt
    - fall
    - slip
    - trip
    - burn
    - broken bone
    - concussion
    - negligence
    - pain
    - suffering
    - medical bills
    - lost wages
    - whiplash
    - liability
    - car accident
    - car wreck
    - car crash
    - motor vehicle accident
    - motorcycle
    - ice
    - dog
    - pitbull
    - rotweiler
    - attack
    - machine
    - malfunction
    - forklift

Case Type is "Mass Torts", When multiple plaintiffs file claims against one or more defendants for harm caused by defective products, pharmaceuticals, or large-scale accidents.
   - Keywords look for are:
     - talc
     - baby powder
     - mesothelioma
     - ovarian cancer
     - hip replacement
     - knee replacement
     - cancer
     - leukemia
     - b-cell
     - t-cell
     - roundup
     - pesticide
     - agriculture
     - non-hodgkins
     - monsanto
     - mesh
     - transvaginal
     - bladder sling
     - zantac
     - valsartan
     - ranitidine
     - paraquat
     - parkinsons
     - lewy
     - CPAP
     - ABAP
     - BiPAP
     - ventilator
     - kidney disease
     - lung disease
     - reactive airway
     - acute respiratory distress
     - multiple myeloma
     - elmiron
     - macular
     - lejeune
     - longshoreman
     - hair relaxer
     - SJS
     - Stevens-Johnson Syndrome
     - Toxic Epidermal Necrolysis
     - Heparin
     - silicosis
     - Hypoxic-ischemic encephalopathy

Case Type is "Nursing Home", When claims involve abuse, neglect, or substandard care in nursing facilities, seeking compensation for harm to elderly or vulnerable residents.
   - Keywords look for are:
     - neglect
     - bed sore
     - pressure sore
     - abuse
     - mistreatment
     - pressure ulcer
     - malnutrition
     - overmedication
     - unexplained injuries
     - understaffed

Case Type is "No matching case type found", When the description contains primarily special characters, symbols, mathematical notations, incomplete words, or valid words that do not clearly relate to legal issues or legal case-related keywords, and if the input is too vague, unclear, or lacks sufficient detail or keywords to determine a case type

Case Rating is depends on severity of an injury. Tier 5 is severe/major injury while Tier 1 is minor injury.
        Case Rating for various case types is given below, use that information for case ratings:
            For Case Type: "Personal Injury"/"Workers Compensation":
                - Tier 1: Minor Injury, Minor dislocation, Minor fracture
                - Tier 2: Sprain, strain, whiplash, contusions, bruises, medical treatment, medication, physical therapy treatment, tingling, numbing sensations.
                - Tier 3: Broken bones etc. with no surgery, Injections, Concussion
                - Tier 4: Surgery or Scheduled surgery, Memory loss
                - Tier 5: Amputation of body parts other than finger or toe, Multiple Surgeries, Crush, Electrocuted, Death, suicide,Machine malfunction with severe injuries, Semitruck accident with surgery
                Note: Any accident that involves a semitruck tracks the case up 1 tier

            For Case Type "Nursing Home":
                - Tier 2: Broken bones or any other injury with no surgery, Malnutrition
                - Tier 3: Surgery or Death
                - Tier 4: Stage 3 or 4 Bedsores

            For Case Type: "Personal Injury":
                - Tier 2: Bleeding, Swelling, laceration, Puncture wounds on extremities with just an antibiotic shot
                - Tier 3: If Multiple bites mentioned, Severe injuries because bites but no surgeries
                - Tier 4: Surgery Or scheduled surgery
                - Tier 5: Plastic surgery to face

            For Case Type: "Personal Injury"/"Medical Malpractice":
                - Tier 3 - Revision surgery is needed
                - Tier 4 - Multiple revision surgeries, Lasting issues as a result of the surgery or misdiagnoses
                - Tier 5 - Unexpected Death as a result of a surgery that wasn't at a high risk of death

            For Case Type "Employment Law":
                - Tier 2: Minor workplace harassment or discrimination, retaliation without significant career impact
                - Tier 3: Wrongful termination or significant harassment/retaliation (requiring psychological counseling)
                - Tier 4: Long-term career loss or ongoing harassment/discrimination (resulting in disability or severe psychological harm)
                - Tier 5: Severe career damage (such as loss of a professional license), or prolonged harassment/retaliation resulting in physical injury or suicide attempt

            For Case Type "Product Liability":
                - Tier 2: Minor injuries from product malfunction (e.g., minor cuts, burns)
                - Tier 3: Moderate injuries requiring medical attention or hospitalization (e.g., moderate burns, fractures)
                - Tier 4: Severe injuries (e.g., disfigurement, loss of function) requiring long-term medical treatment or surgeries
                - Tier 5: Death or catastrophic injuries (e.g., permanent disability) due to product malfunction or failure to warn

            For Case Type "Mass Torts":
                - Tier 2: Minor side effects or medical issues linked to product use (e.g., mild allergic reactions)
                - Tier 3: Moderate medical conditions (e.g., respiratory distress, moderate organ damage) from prolonged exposure or use of the product
                - Tier 4: Severe and lasting conditions (e.g., cancer, severe organ damage) requiring extensive treatment or surgeries
                - Tier 5: Death or severe life-altering conditions (e.g., mesothelioma, severe cancer cases, multiple organ failure)

If the description contains primarily special characters, symbols, mathematical notations, incomplete words, or valid words that do not clearly relate to legal issues or legal case-related keywords, and if the input is too vague,
unclear, or lacks sufficient detail to determine a case type, respond with:
    "Status": "NO"
    "CaseType": "No matching case type found"
    "Case_Rating": "case rating cannot be determined. please provide additional information"
    "Explanation": Provide a detailed reason explaining why the description is insufficient and suggest the user provide more specific or relevant legal case information.

If the description aligns with what is understood as a legal case issue, then analyze the description. For each Case Type if keywords match then respond with one of these case type only - Workers Compensation,Medical Malpractice,Employment Law,Product Liability,Personal Injury,Mass Torts,Nursing Home, and If no keywords match then dont respond with any Case Type other than given 7 and just respond with:
    "CaseType": "No matching case type found"

If the severity of the description does not match with case rating criteria, respond with:
    "Case_Rating": "case rating cannot be determined. please provide additional information",

The Output should be strictly in correct JSON format without json keyword and the JSON structure must have the following key values:
    "Status": "YES or NO (say YES if the description is proper legal case description or related to legal issue, otherwise say NO)",
    "CaseType": "Case Type here",
    "Case_Rating": "Case Rating here",
    "Explanation": "Explain your answer here with detail reason behind case, why?

  """
      
    def load_llm(self):
        openai.api_type = "azure"
        openai.api_key = os.getenv("OPENAI_API_KEY")
        openai.api_version = os.getenv('OPENAI_DEPLOYMENT_VERSION')
        llm = AzureChatOpenAI(
            deployment_name=self.OPENAI_DEPLOYMENT_NAME,
            model_name=self.OPENAI_MODEL_NAME,
            azure_endpoint=self.OPENAI_DEPLOYMENT_ENDPOINT,
            openai_api_version=self.OPENAI_DEPLOYMENT_VERSION,
            openai_api_key=self.OPENAI_API_KEY,
            openai_api_type="azure"
            )

        return llm

    def get_predictions(self, prompt_hf):
        llm = self.load_llm()
        predictions = llm.predict(prompt_hf)
        return predictions

    def analyze_case(self, query):
        try:
            hf_prompt = self.custom_prompt_template.format(question=query)
            predictions = self.get_predictions(hf_prompt)
            return predictions
        except Exception as error:
            print(error)
            return None  # Unable to determine the status from LLM


def flag_check(query):
    case_classifier = caseClassifier()
    app = caseClassifierApp(case_classifier)
    response = app.send(query)
    pred = json.loads(response)
    #print("------------------------------>>>>",type(pred))
    return pred
    #response = get_string_between_braces(response)
    # try:
    #     #print("*************************************************")
    #     #print(response)
    #     #print("*************************************************")
    #     pred = json.loads(response)
    #     expl = pred.get("Explanation", "").strip()
    #     #print(expl)
    #     try:
    #         flag = pred.get("Status", "").strip().upper() == "YES"
    #         if flag is True:
    #             logging.info("This appears to be a legal case.")
    #             final_result = process_query(query)
    #             #return final_result
    #         elif flag is False:
    #             #print ("Flag is NO")
    #             logging.info("This does not appear to be a legal case.")
    #             final_result = '''{{
    #                         "PrimaryCaseType": " ",
    #                         "SecondaryCaseType": " ",
    #                         "CaseRating": " ",
    #                         "CaseState" : " ",
    #                         "IsWorkersCompensation (Yes/No)?": " ",
    #                         "Confidence(%)": " ",
    #                         "Explanation": "{e}",
    #                         "Handling Firm" : "Unknown"
    #                     }}'''.format(e=expl)
    #            # print (final_result)

    #             #return final_result
    #         else:
    #             logging.warning("Unable to determine if it's a legal case due to an unexpected response.")
    #             final_result = '''
    #                 {
    #                     "PrimaryCaseType": " ",
    #                     "SecondaryCaseType": " ",
    #                     "CaseRating": " ",
    #                     "CaseState" : " ",
    #                     "IsWorkersCompensation (Yes/No)?": " ",
    #                     "Confidence(%)": " ",
    #                     "Explanation": "There is some error occured while answering your question, Please try with same case description again.  Sorry for an inconvenience Caused",
    #                     "Handling Firm" : "Unknown"
    #                 }
    #                 '''
    #             #return final_result
    #         return final_result
    #     except Exception as error:
    #         logging.exception("An error occurred in flag_check: %s", error)
    # except Exception as error:
    #     logging.exception("An error occurred in flag_check: %s", error)


class caseClassifierApp:
    def __init__(self, case_classifier):
        self.case_classifier = case_classifier

    def send(self, msg):
        result = self.case_classifier.analyze_case(msg)
        return result


def get_string_between_braces(text):
    start = text.find('{')
    end = text.rfind('}')
    if start != -1 and end != -1 and start < end:
        return text[start:end+1]
    else:
        return None


#if __name__ == "__main__":
 #   while True:
  #      query = input('you: ')
   #     if query == 'q':
    #        break
     #   elif query.strip() == "":
      #      continue
        #response = app.send(query)
       # response = flag_check(query)
        #logging.info('Final result generated: %s', response)
       # print("response", response)













