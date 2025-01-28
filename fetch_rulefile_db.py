import sqlite3
import json
import logging

def fetch_data_by_casestate(case_state):
    
    try:
        conn = sqlite3.connect('Cases.db')
        curr = conn.cursor()
        try:
            # fetch and map ID from CaseStates
            curr.execute("SELECT CaseStateId FROM CaseStates WHERE Name = ?", (case_state,))
            result = curr.fetchone()
            case_state_id = result[0]
            if result is not None:
                case_state_id = result[0]
                curr.execute("SELECT Rules FROM Case_rules WHERE CaseStateId = ?", (case_state_id,))
                result = curr.fetchone()
                rule_state = result[0]
                data = json.loads(rule_state)

                return data
                # for rule in data["rules"]:
                #     self.hf_prompt_template += f"  - If the case rating is '{rule['condition']['case_rating']}' and case type is '{rule['condition']['case_type']}', {rule['action']}\n"
                    
                # print(hf_prompt_template)

            else:
                qa_result["Case State"] = "Unknown"
                hf_result = '''
                {
                    "Handling Firm" : "SAD"
                }
                '''
                return hf_result
            
        
        except Exception as error:
            print(error)
            logging.error('An error occurred while reading firm rules file: %s', error)

            hf_result = '''
            {
                "Handling Firm" : "SAD"
            }
            '''
            return hf_result
        
      
    except IOError as e:
        # Log an error if there is an issue reading the firm rules file
        logging.error('An error occurred while reading firm rules file: %s', e)
        hf_result = '''
            {
                "Handling Firm" : "SAD"
            }
            '''
        return hf_result

if __name__ == "__main__":

    qa_result = {"PrimaryCaseType": "Wrongful Death", "SecondaryCaseType": "Employment Law", "CaseRating": "Tier 5", "Case State": "CA California", "Is Workers Compensation (Yes/No)?": "No", "Confidence(%)": "95%", "Explanation": "The description mentions that the client's father was killed due to the negligence of his employer on a construction site. The employers went to great lengths to cover up the circumstances and are multi-millionaires expected to fight the case aggressively. Based on this, the primary case type is wrongful death, and the secondary case type is employment law. The case rating is Tier 5 since the client's father died. The incident happened on a construction site, which is not the client's workplace, so it is not a workers' compensation case. The description mentions the location as Sonoma County, CA, so the case state is CA California.\n\nNo handling firm available for Tier 5 and Wrongful Death or Employment Law case type in California","CaseId": "3cd87157-4de4-4ae1-8f9f-8947c66c615d"}
    case_state = qa_result.get("Case State")
    # Extract the state abbreviation from the case state
    if " " in case_state:
        state_parts = case_state.split(" ", 1)
        if not state_parts[0].isupper():
            case_state = state_parts[0] + " " + state_parts[1]
        else:
            case_state = state_parts[1]
    else:
        case_state = case_state

    data = fetch_data_by_casestate(case_state)
    print("data.......",data)
