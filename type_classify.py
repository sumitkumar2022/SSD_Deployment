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
        # DefaultAzureCredential will handle authentication for managed identity, Azure CLI, and environment variables.
        #credential = DefaultAzureCredential()
        credential = AzureCliCredential()
        # Create a SecretClient using the Key Vault URL and credential
        client = SecretClient(vault_url=key_vault_url, credential=credential)
        # Retrieve a secret
        #self.OPENAI_API_KEY = client.get_secret("pl-open-api-key").value
        #self.OPENAI_DEPLOYMENT_VERSION = client.get_secret("pl-openai-deployment-version").value
        #self.OPENAI_DEPLOYMENT_ENDPOINT = client.get_secret("pl-openai-deployment-endpoint").value
        #self.OPENAI_DEPLOYMENT_NAME = client.get_secret("pl-openai-deployment-name").value
        #self.OPENAI_MODEL_NAME = client.get_secret("pl-openai-model").value
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

        self.custom_prompt_template = """
        
        Given the following legal case description, please determine the most suitable case type for the given description: "{question}" 
        Classify this legal case into one of the following potential case types based on the presence of relevant keywords:
        1. **Workers Compensation**:
            - Keywords: injured at work, workplace injury, workers' comp, on-the-job, occupational injury, compensation, work-related injury

        2. **Medical Malpractice**:
            - Keywords: misdiagnosis, malpractice, wrong treatment, surgical error, medication error, hospital, doctor, nurse, improper care, birth injury, d               elayed diagnosis, incorrect prescription 
        3. **Employment Law**:
            - Keywords: discrimination, harassment, wrongful termination, fired, laid off, hostile work environment, retaliation, sexual harassment, unequal pay, whistleblower, race, gender, disability

        4. **Product Liability**:
            - Keywords: defective product, malfunction, injury from product, unsafe, recall, failure to warn, design defect, manufacturing defect, consumer safety

        5. **Personal Injury**:
            - Keywords: accident, injury, hurt, fall, slip, trip, burn, broken bone, concussion, negligence, pain, suffering, medical bills, lost wages, whiplash, liability, car accident, car wreck, 
    car crash, motor vehicle accident, motorcycle, ice, dog, pitbull, rotweiler, attack, machine, malfunction, forklift

        6. **Mass Torts**:
            - Keywords: talc, baby powder, mesothelioma, ovarian cancer, hip replacement, knee replacement, cancer, leukemia, b-cell, t-cell, roundup, pesticide, agriculture, non-hodgkins, monsanto, 
    mesh, transvaginal, bladder sling, zantac, valsartan, ranitidine, paraquat, parkinsons, lewy, CPAP, ABAP, BiPAP, ventilator, kidney disease, lung disease, reactive airway, acute respiratory 
    distress, multiple myeloma, elmiron, macular, lejeune, longshoreman, hair relaxer, SJS, Stevens-Johnson Syndrome, Toxic Epidermal Necrolysis, Heparin, silicosis, Hypoxic-ischemic encephalopathy

        7. **Nursing Home**:
            - Keywords: neglect, bed sore, pressure sore, abuse, mistreatment, pressure ulcer, malnutrition, overmedication, unexplained injuries, understaffed

        Analyze the legal case description, search for relevant keywords, and classify it into the most appropriate potential claim type.
        If no keywords match, respond with "No matching case type found."

        The output should be in the following JSON format without the json keyword and the JSON structure must have the following key values:
        {
            "CaseType" : "Case Type here",
            "Explanation" : "Explain your answer here with detailed reasoning behind the case and why it falls under this category."
        }


        """

    def load_llm(self):
        openai.api_type = "azure"
        openai.api_key = os.getenv("OPENAI_API_KEY")
        openai.api_version = os.getenv('OPENAI_DEPLOYMENT_VERSION')
        llm = AzureChatOpenAI(deployment_name=self.OPENAI_DEPLOYMENT_NAME,
                              model_name=self.OPENAI_MODEL_NAME,
                              azure_endpoint=self.OPENAI_DEPLOYMENT_ENDPOINT,
                              openai_api_version=self.OPENAI_DEPLOYMENT_VERSION,
                              openai_api_key=self.OPENAI_API_KEY,
                              openai_api_type="azure")


        return llm
    
    def get_predictions(self, prompt_hf):
        llm = self.load_llm()
        predictions = llm.predict(prompt_hf)
        print(predictions)
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
    #response = get_string_between_braces(response)
    try:
        #print("*************************************************")
        #print(response)
        #print("*************************************************")
        final_result = json.loads(response)
        #expl = pred.get("Explanation", "").strip()
        #print(expl)
        #try:
            #final_result = json.loads(response)
            #flag = pred.get("Status", "").strip().upper() == "YES"
            #if flag is True:
            #logging.info("This appears to be a legal case.")
            #final_result = process_query(query)
            #return final_result
            #print("description is legal")
            #elif flag is False:
                #print ("Flag is NO")
                #logging.info("This does not appear to be a legal case.")
                #final_result = '''{{
                            #"PrimaryCaseType": " ",
                            #"SecondaryCaseType": " ",
                            #"CaseRating": " ",
                            #"Case State" : " ",
                            #"Is Workers Compensation (Yes/No)?": " ",
                            #"Confidence(%)": " ",
                            #"Explanation": "{e}",
                            #"Handling Firm" : "Unknown"
                        #}}'''.format(e=expl)
               # print (final_result)
                        
                #return final_result
            #else:
                #logging.warning("Unable to determine if it's a legal case due to an unexpected response.")
                #final_result = '''
                    #{
                        #"PrimaryCaseType": " ",
                        #"SecondaryCaseType": " ",
                        #"CaseRating": " ",
                        #"Case State" : " ",
                        #"Is Workers Compensation (Yes/No)?": " ",
                        #"Confidence(%)": " ",
                        #"Explanation": "There is some error occured while answering your question, Please try with same case description again.  Sorry for an inconvenience Caused",
                        #"Handling Firm" : "Unknown"
                    #}
                    #'''
                #return final_result
        return final_result
        #except Exception as error:
            #logging.exception("An error occurred in flag_check: %s", error)
    except Exception as error:
        logging.exception("An error occurred in flag_check: %s", error)  


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


if __name__ == "__main__":

    while True:
        query = input('you: ')
        if query == 'q':
            break
        elif query.strip() == "":
            continue
        #response = app.send(query)
        response = flag_check(query)
        logging.info('Final result generated: %s', response)
        print("response", response)
        
