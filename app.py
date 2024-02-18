import os

import numpy as np
import openai
import pandas as pd
from dotenv import load_dotenv
from flask import Flask, render_template, request
from sklearn.metrics.pairwise import cosine_similarity

load_dotenv()

openai.api_key = os.getenv("openaitoken")

app = Flask(__name__)

def process_symptoms(symptoms):
    """returns list of potential diseases"""
    symptoms_list = ['pain chest', 'shortness of breath', 'dizziness', 'asthenia', 'fall', 'syncope', 'vertigo', 'sweat/sweating increased', 'palpitation', 'nausea', 'angina pectoris', 'pressure chest', 'polyuria', 'polydypsia', 'orthopnea', 'rale', 'unresponsiveness', 'mental status changes', 'vomiting', 'labored breathing', 'feeling suicidal', 'suicidal', 'hallucinations auditory', 'feeling hopeless', 'weepiness', 'sleeplessness', 'motor retardation', 'irritable mood', 'blackout', 'mood depressed', 'hallucinations visual', 'worry', 'agitation', 'tremor', 'intoxication', 'verbal auditory hallucinations', 'energy increased', 'difficulty', 'nightmare', 'unable to concentrate', 'homelessness', 'hypokinesia', 'dyspnea on exertion', 'chest tightness', 'cough', 'fever', 'decreased translucency', 'productive cough', 'pleuritic pain', 'yellow sputum', 'breath sounds decreased', 'chill', 'rhonchus', 'green sputum', 'non-productive cough', 'wheezing', 'haemoptysis', 'distress respiratory', 'tachypnea', 'malaise', 'night sweat', 'jugular venous distention', 'dyspnea', 'dysarthria', 'speech slurred', 'facial paresis', 'hemiplegia', 'seizure', 'numbness', 'symptom aggravating factors', 'st segment elevation', 'st segment depression', 't wave inverted', 'presence of q wave', 'chest discomfort', 'bradycardia', 'pain', 'nonsmoker', 'erythema', 'hepatosplenomegaly', 'pruritus', 'diarrhea', 'abscess bacterial', 'swelling', 'apyrexial', 'dysuria', 'hematuria', 'renal angle tenderness', 'lethargy', 'hyponatremia', 'hemodynamically stable', 'difficulty passing urine', 'consciousness clear', 'guaiac positive', 'monoclonal', 'ecchymosis', 'tumor cell invasion', 'haemorrhage', 'pallor', 'fatigue', 'heme positive', 'pain back', 'orthostasis', 'arthralgia', 'transaminitis', 'sputum purulent', 'hypoxemia', 'hypercapnia', 'patient non compliance', 'unconscious state', 'bedridden/bedridden', 'abdominal tenderness', 'unsteady gait', 'hyperkalemia', 'urgency of micturition', 'ascites', 'hypotension', 'enuresis', 'asterixis', 'muscle twitch', 'sleepy', 'headache', 'lightheadedness', 'food intolerance', 'numbness of hand', 'general discomfort', 'drowsiness', 'stiffness', 'prostatism', 'drowsiness/sleepy', 'weight gain', 'fatigue/tired', 'mass of body structure', 'has religious belief', 'nervousness', 'formication', 'hot flush', 'lesion', 'cushingoid facies/cushingoid habitus', 'emphysematous change', 'decreased body weight', 'hoarseness', 'thicken', 'spontaneous rupture of membranes', '', 'muscle hypotonia/hypotonic', 'redness', 'hypesthesia', 'hyperacusis', 'scratch marks', 'sore to touch', 'burning sensation', 'satiety early', 'throbbing sensation quality', 'sensory discomfort', 'constipation', 'pain abdominal', 'heartburn', 'breech presentation', 'cyanosis', 'pain in lower limb', 'cardiomegaly', 'clonus', 'unwell', 'anorexia', 'syncope/blackout/history of - blackout', 'anosmia', 'metastatic lesion', 'hemianopsia homonymous', 'hematocrit decreased', 'neck stiffness', 'cicatrisation', 'hypometabolism', 'aura', 'myoclonus', 'gurgle', 'wheelchair bound', 'left atrial hypertrophy', 'oliguria', 'catatonia', 'unhappy', 'paresthesia', 'gravida 0', 'lung nodule', 'distended abdomen', 'ache', 'macerated skin', 'heavy feeling', 'rest pain', 'sinus rhythm', 'withdraw', 'behavior hyperactive', 'terrify', 'photopsia', 'giddy mood', 'disturbed family', 'hypersomnia', 'hyperhidrosis disorder', 'mydriasis', 'extrapyramidal sign', 'loose associations', 'exhaustion', 'snore', 'r wave feature', 'overweight', 'systolic murmur', 'asymptomatic', 'splenomegaly', 'bleeding of vagina', 'macule', 'photophobia', 'painful swallowing', 'cachexia', 'hypocalcemia result', 'hypothermia, natural', 'atypia', 'general unsteadiness', 'throat sore', 'snuffle', 'hacking cough', 'stridor', 'paresis', 'aphagia', 'focal seizures', 'abnormal sensation', 'stupor', 'fremitus', "Stahli's line", 'stinging sensation', 'paralyse', 'hirsutism', 'sniffle', 'bradykinesia', 'out of breath', 'urge incontinence', 'vision blurred', 'room spinning', 'rambling speech', 'clumsiness', 'decreased stool caliber', 'hematochezia', 'egophony', 'cicatrisation/scar tissue', 'neologism', 'decompensation', 'stool color yellow', 'rigor - temperature-associated observation', 'paraparesis', 'moody', 'fear of falling', 'spasm', 'hyperventilation', 'excruciating pain', 'gag', 'posturing', 'pulse absent', 'dysesthesia', 'polymyalgia', 'passed stones', 'qt interval prolonged', 'ataxia', "Heberden's node", 'hepatomegaly', 'sciatica', 'frothy sputum', 'mass in breast', 'retropulsion', 'estrogen use', 'hypersomnolence', 'underweight/underweight', 'dullness', 'red blotches', 'colic abdominal', 'hypokalemia', 'hunger', 'prostate tender', 'pain foot', 'urinary hesitation', 'disequilibrium', 'flushing', 'indifferent mood', 'urinoma', 'hypoalbuminemia', 'pustule', 'slowing of urinary stream', 'extreme exhaustion', 'no status change', 'breakthrough pain', 'pansystolic murmur', 'systolic ejection murmur', 'stuffy nose', 'barking cough', 'rapid shallow breathing', 'noisy respiration', 'nasal discharge present', 'frail', 'cystic lesion', 'projectile vomiting', 'heavy legs', 'titubation', 'dysdiadochokinesia', 'achalasia', 'side pain', 'monocytosis', 'posterior rhinorrhea', 'incoherent', 'lameness/claudication', 'clammy skin', 'mediastinal shift', 'nausea and vomiting', 'awakening early', 'tenesmus', 'fecaluria', 'pneumatouria', 'todd paralysis', 'alcoholic withdrawal symptoms', 'myalgia', 'dyspareunia', 'poor dentition', 'floppy', 'inappropriate affect', 'poor feeding', 'moan', 'welt', 'tinnitus', 'hydropneumothorax', 'superimposition', 'feeling strange', 'uncoordination', 'absences finding', 'tonic seizures', 'debilitation', 'impaired cognition', 'drool', 'pin-point pupils', 'tremor resting', 'groggy', 'adverse reaction/adverse effect', 'abdominal bloating', 'fatigability', 'para 2', 'abortion', 'intermenstrual heavy bleeding', 'previous pregnancies 2', 'primigravida', 'abnormally hard consistency', 'proteinemia', 'pain neck', 'dizzy spells', 'shooting pain', 'hyperemesis', 'milky', 'regurgitates after swallowing', 'lip smacking', 'phonophobia', 'rolling of eyes', 'ambidexterity', 'pulsus paradoxus', 'gravida 10', 'bruit', 'breath-holding spell', 'scleral icterus', 'retch', 'blanch', 'elation', 'verbally abusive behavior', 'transsexual', 'behavior showing increased motor activity', 'scar tissue', 'coordination abnormal', 'choke', 'bowel sounds decreased', 'no known drug allergies', 'low back pain', 'charleyhorse', 'sedentary', 'feels hot/feverish', 'flare', 'pericardial friction rub', 'hoard', 'panic', 'cardiovascular finding/cardiovascular event', 'soft tissue swelling', 'rhd positive', 'para 1', 'nasal flaring', 'sneeze', 'hypertonicity', "Murphy's sign", 'flatulence', 'gasping for breath', 'feces in rectum', 'prodrome', 'hypoproteinemia', 'alcohol binge episode', 'abdomen acute', 'air fluid level', 'catching breath', 'large-for-dates fetus', 'immobile', 'homicidal thoughts']
    symptoms_df = pd.read_csv('data/disease_symptoms_encoded.csv')
    symptom_columns = symptoms_df.columns[:-1]  # Exclude the last column, which is 'Disease'
    # Convert symptom columns to arrays for each disease
    bit_vectors = symptoms_df[symptom_columns].to_numpy()

    # generate one hot encoded features for the test illness
    test_illness_df = pd.DataFrame(columns=symptoms_list)
    test_illness_dict = {symptom: 0 for symptom in symptoms_list}
    for symptom in symptoms:
        if symptom in symptoms_list:
            test_illness_dict[symptom] = 1
    test_illness_df.loc[0]=(pd.Series(test_illness_dict))


    # create a dataframe from the dict
    test_illness_bit_vector = test_illness_df[symptom_columns].to_numpy()
    cosine_similarities = cosine_similarity(test_illness_bit_vector, bit_vectors)
    top_3_indices = np.argsort(cosine_similarities[0])[-3:][::-1]
    disease_values = np.array(['hypertensive  disease', 'diabetes', 'depression  mental/depressive disorder', 'coronary  arteriosclerosis/coronary heart disease', 'pneumonia', 'failure  heart congestive', 'accident  cerebrovascular', 'asthma', 'myocardial  infarction', 'hypercholesterolemia', 'infection', 'infection  urinary tract', 'anemia', 'chronic  obstructive airway disease', 'dementia', 'insufficiency  renal', 'confusion', 'degenerative  polyarthritis', 'hypothyroidism', 'anxiety  state', 'malignant  neoplasms/primary malignant neoplasm', 'acquired  immuno-deficiency  syndrome/HIV/hiv infections', 'cellulitis', 'gastroesophageal  reflux disease', 'septicemia/systemic  infection/sepsis (invertebrate)', 'deep  vein thrombosis', 'dehydration', 'neoplasm', 'embolism  pulmonary', 'epilepsy', 'cardiomyopathy', 'chronic  kidney failure', 'carcinoma', 'hepatitis  C', 'peripheral  vascular disease', 'psychotic  disorder', 'hyperlipidemia', 'bipolar  disorder', 'obesity', 'ischemia', 'cirrhosis', 'exanthema', 'benign  prostatic hypertrophy', 'kidney  failure acute', 'mitral  valve insufficiency', 'arthritis', 'bronchitis', 'hemiparesis', 'osteoporosis', 'transient  ischemic attack', 'adenocarcinoma', 'paranoia', 'pancreatitis', 'incontinence', 'paroxysmal  dyspnea', 'hernia', 'malignant  neoplasm of prostate/carcinoma prostate', 'edema  pulmonary', 'lymphatic  diseases', 'stenosis  aortic valve', 'malignant  neoplasm of breast/carcinoma breast', 'schizophrenia', 'diverticulitis', 'overload  fluid', 'ulcer  peptic', 'osteomyelitis', 'gastritis', 'bacteremia', 'failure  kidney', 'sickle  cell anemia', 'failure  heart', 'upper  respiratory infection', 'hepatitis', 'hypertension  pulmonary', 'deglutition  disorder', 'gout', 'thrombocytopaenia', 'hypoglycemia', 'pneumonia  aspiration', 'colitis', 'diverticulosis', 'suicide  attempt', 'Pneumocystis  carinii pneumonia', 'hepatitis  B', 'parkinson  disease', 'lymphoma', 'hyperglycemia', 'encephalopathy', 'tricuspid  valve insufficiency', "Alzheimer's  disease", 'candidiasis/oral  candidiasis', 'neuropathy', 'kidney  disease', 'fibroid  tumor', 'glaucoma', 'neoplasm  metastasis', 'malignant  tumor of colon/carcinoma colon', 'ketoacidosis  diabetic', 'tonic-clonic  epilepsy/tonic-clonic seizures', 'malignant  neoplasms', 'respiratory  failure', 'melanoma', 'gastroenteritis', 'malignant  neoplasm of lung/carcinoma of lung', 'manic  disorder', 'personality  disorder', 'primary  carcinoma of the liver cells', 'emphysema  pulmonary', 'hemorrhoids', 'spasm  bronchial', 'aphasia', 'obesity  morbid', 'pyelonephritis', 'endocarditis', 'effusion  pericardial/pericardial effusion body substance', 'chronic  alcoholic intoxication', 'pneumothorax', 'delirium', 'neutropenia', 'hyperbilirubinemia', 'influenza', 'dependence', 'thrombus', 'cholecystitis', 'hernia  hiatal', 'migraine  disorders', 'pancytopenia', 'cholelithiasis/biliary  calculus', 'tachycardia  sinus', 'ileus', 'adhesion', 'delusion', 'affect  labile', 'decubitus  ulcer'])

    diseases = disease_values[top_3_indices]
    # capitalize first letter for each disease
    diseases = [disease.capitalize() for disease in diseases]
    return list(diseases)

def gpt_symptoms(symptoms):
    '''returns a list of potential diseases from the GPT model'''

    # return "Lorem ipsum dolor sit amet, consectetur adipiscing elit. Nulla nec odio nec nunc tincidunt tincidunt\n1. fizz \n2. buzz \n3. bear"

    file = open("data/disease_symptom.csv", "r")
    all_text = file.read()
    messages = [ {"role": "system", "content":  
                "You are given the following data in a .csv file \n" + all_text + ". The user will list multiple symptoms. Please list at most 5 diseases, and record their all symptoms from the file in a comma list. Provide a basic response with the other matched symptoms and dont mention the file"} ] 

    message = str(symptoms)

    if message: 
        messages.append( 
            {"role": "user", "content": message}, 
        ) 
        chat = openai.ChatCompletion.create( 
            model="gpt-3.5-turbo", messages=messages 
        ) 
    reply = chat.choices[0].message.content 
    # print(chat)
    return reply

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Retrieve symptoms from the form
        symptoms = request.form.get('symptoms')
        
        # Split symptoms into a list
        symptoms_list = [symptom.strip() for symptom in symptoms.split(',')]
        
        # Process the symptoms
        results = process_symptoms(symptoms_list)
        gpt = gpt_symptoms(symptoms_list)

        # Render the results template with the processed results
        return render_template('symptom.html', symptoms=symptoms, gpt=gpt, results=results)
    
    # Render the index template for GET requests
    return render_template('about.html')

@app.route('/symptom')
def about():
    # get symptoms from data/symptoms.txt
    results=[]
    with open('data/symptoms.txt', 'r') as file:
        symptoms = file.readlines()
    return render_template('symptom.html', symptoms=[], gpt="", results=results)

if __name__ == '__main__':
    app.run(debug=True)
