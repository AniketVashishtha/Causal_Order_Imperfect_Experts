"""
Benchmark causal graph definitions used in the paper.

Each graph is stored as a dictionary with the following keys:
  - nodes: list of node names
  - ground_truth_edges: list of (source, target) tuples representing the true DAG
  - descriptions: dict mapping node name -> description (optional, may be None)
  - context: string describing the modelling context for LLM prompts
"""


# ---------------------------------------------------------------------------
# Cancer (5 nodes)
# ---------------------------------------------------------------------------
CANCER = {
    "nodes": ['smoker', 'pollution', 'cancer', 'xray', 'dyspnoea'],

    "ground_truth_edges": [
        ('smoker', 'cancer'),
        ('pollution', 'cancer'),
        ('cancer', 'xray'),
        ('cancer', 'dyspnoea'),
    ],

    "descriptions": {
        'smoker': 'smoking habit',
        'pollution': 'exposure to pollutants',
        'cancer': 'cancer',
        'xray': 'getting positive xray result',
        'dyspnoea': 'dyspnoea',
    },

    "context": "model the relation between various variables responsible for causing Cancer and its possible outcomes",
}


# ---------------------------------------------------------------------------
# Asia (8 nodes)
# ---------------------------------------------------------------------------
ASIA = {
    "nodes": [
        'visit to Asia', 'tuberculosis', 'either tuberculosis or lung cancer',
        'positive X-ray', 'dyspnoea', 'bronchitis', 'lung cancer', 'smoking',
    ],

    "ground_truth_edges": [
        ('visit to Asia', 'tuberculosis'),
        ('tuberculosis', 'either tuberculosis or lung cancer'),
        ('either tuberculosis or lung cancer', 'positive X-ray'),
        ('smoking', 'lung cancer'),
        ('smoking', 'bronchitis'),
        ('lung cancer', 'either tuberculosis or lung cancer'),
        ('either tuberculosis or lung cancer', 'dyspnoea'),
        ('bronchitis', 'dyspnoea'),
    ],

    "descriptions": {
        'visit to Asia': 'visiting Asian countries with high exposure to pollutants',
        'tuberculosis': 'tuberculosis',
        'either tuberculosis or lung cancer': 'tuberculosis or lung cancer',
        'positive X-ray': 'getting positve xray result',
        'dyspnoea': 'dyspnoea',
        'bronchitis': 'bronchitis',
        'lung cancer': 'lung cancer',
        'smoking': 'smoking habit',
    },

    "context": "model the possible respiratory problems someone can have who has recently visited Asia and is experiencing shortness of breath",
}


# ---------------------------------------------------------------------------
# Child (20 nodes)
# ---------------------------------------------------------------------------
CHILD = {
    "nodes": [
        'BirthAsphyxia', 'HypDistrib', 'HypoxiaInO2', 'CO2', 'ChestXray',
        'Grunting', 'LVHreport', 'LowerBodyO2', 'RUQO2', 'CO2Report',
        'XrayReport', 'Disease', 'GruntingReport', 'Age', 'LVH', 'DuctFlow',
        'CardiacMixing', 'LungParench', 'LungFlow', 'Sick',
    ],

    "ground_truth_edges": [
        ('BirthAsphyxia', 'Disease'),
        ('HypDistrib', 'LowerBodyO2'),
        ('HypoxiaInO2', 'LowerBodyO2'),
        ('HypoxiaInO2', 'RUQO2'),
        ('CO2', 'CO2Report'),
        ('ChestXray', 'XrayReport'),
        ('Grunting', 'GruntingReport'),
        ('Disease', 'Age'),
        ('Disease', 'LVH'),
        ('Disease', 'DuctFlow'),
        ('Disease', 'CardiacMixing'),
        ('Disease', 'LungParench'),
        ('Disease', 'LungFlow'),
        ('Disease', 'Sick'),
        ('LVH', 'LVHreport'),
        ('DuctFlow', 'HypDistrib'),
        ('CardiacMixing', 'HypDistrib'),
        ('CardiacMixing', 'HypoxiaInO2'),
        ('LungParench', 'HypoxiaInO2'),
        ('LungParench', 'CO2'),
        ('LungParench', 'ChestXray'),
        ('LungParench', 'Grunting'),
        ('LungFlow', 'ChestXray'),
        ('Sick', 'Grunting'),
        ('Sick', 'Age'),
    ],

    "descriptions": {
        'BirthAsphyxia': "lack of oxygen to the blood during the infant's birth",
        'HypDistrib': "low oxygen areas equally distributed around the body",
        'HypoxiaInO2': "hypoxia when breathing oxygen",
        'CO2': "level of carbon dioxide in the body",
        'ChestXray': "having a chest x-ray",
        'Grunting': "grunting in infants",
        'LVHreport': "report of having left ventricular hypertrophy",
        'LowerBodyO2': "level of oxygen in the lower body",
        'RUQO2': "level of oxygen in the right up quadricep muscule",
        'CO2Report': "a document reporting high level of CO2 levels in blood",
        'XrayReport': "lung excessively filled with blood",
        'Disease': "infant methemoglobinemia",
        'GruntingReport': "report of infant grunting",
        'Age': "age of infant at disease presentation",
        'LVH': "thickening of the left ventricle",
        'DuctFlow': "blood flow across the ductus arteriosus",
        'CardiacMixing': "mixing of oxygenated and deoxygenated blood",
        'LungParench': "the state of the blood vessels in the lungs",
        'LungFlow': "low blood flow in the lungs",
        'Sick': "presence of an illness",
    },

    "context": "model congenital heart disease in babies",
}


# ---------------------------------------------------------------------------
# Covid Respiratory (11 nodes)
# ---------------------------------------------------------------------------
COVID = {
    "nodes": [
        'Virus Enters Upper Respiratory Tract',
        'Upper Respiratory Tract epithelial infection',
        'Infection of olfactory epithelium',
        'Anosmia and/or aguesia',
        'Alveolar epithelial infection',
        'Alveolar endothelial infection',
        'Viremia',
        'Systemic immune/inflammatory response',
        'Pulmonary capillary leakage',
        'Dry cough',
        'Productive cough',
    ],

    "ground_truth_edges": [
        ('Virus Enters Upper Respiratory Tract', 'Upper Respiratory Tract epithelial infection'),
        ('Virus Enters Upper Respiratory Tract', 'Alveolar epithelial infection'),
        ('Upper Respiratory Tract epithelial infection', 'Alveolar epithelial infection'),
        ('Upper Respiratory Tract epithelial infection', 'Infection of olfactory epithelium'),
        ('Upper Respiratory Tract epithelial infection', 'Dry cough'),
        ('Upper Respiratory Tract epithelial infection', 'Systemic immune/inflammatory response'),
        ('Upper Respiratory Tract epithelial infection', 'Viremia'),
        ('Infection of olfactory epithelium', 'Anosmia and/or aguesia'),
        ('Alveolar epithelial infection', 'Productive cough'),
        ('Alveolar epithelial infection', 'Pulmonary capillary leakage'),
        ('Alveolar epithelial infection', 'Systemic immune/inflammatory response'),
        ('Alveolar epithelial infection', 'Viremia'),
        ('Alveolar epithelial infection', 'Alveolar endothelial infection'),
        ('Alveolar endothelial infection', 'Pulmonary capillary leakage'),
        ('Alveolar endothelial infection', 'Systemic immune/inflammatory response'),
        ('Alveolar endothelial infection', 'Viremia'),
        ('Pulmonary capillary leakage', 'Productive cough'),
        ('Pulmonary capillary leakage', 'Dry cough'),
        ('Systemic immune/inflammatory response', 'Pulmonary capillary leakage'),
        ('Viremia', 'Systemic immune/inflammatory response'),
    ],

    "descriptions": None,

    "context": "modeling the initial pathophysiological process of SARS-CoV-2 in the respiratory system involves outlining the various pathways from viral infection to key complications.",
}


# ---------------------------------------------------------------------------
# Alzheimer's (11 nodes)
# ---------------------------------------------------------------------------
ALZHEIMERS = {
    "nodes": [
        'sex', 'age', 'ventricular volume', 'brain volume', 'av45', 'tau',
        'brain MRI', 'slice number', 'education', 'moca', 'APOE4',
    ],

    "ground_truth_edges": [
        ('sex', 'ventricular volume'),
        ('sex', 'brain volume'),
        ('age', 'ventricular volume'),
        ('age', 'brain volume'),
        ('age', 'av45'),
        ('age', 'tau'),
        ('age', 'moca'),
        ('ventricular volume', 'brain MRI'),
        ('brain volume', 'ventricular volume'),
        ('brain volume', 'moca'),
        ('brain volume', 'ventricular volume'),
        ('brain volume', 'brain MRI'),
        ('brain volume', 'moca'),
        ('av45', 'brain volume'),
        ('av45', 'tau'),
        ('tau', 'moca'),
        ('tau', 'ventricular volume'),
        ('tau', 'brain volume'),
        ('slice number', 'brain MRI'),
        ('education', 'moca'),
        ('APOE4', 'av45'),
    ],

    "descriptions": {
        'APOE4': 'Expression level of APOE4 gene',
        'sex': 'Biological Sex of Patient',
        'age': 'Age of Patient',
        'education': 'Educational attainment (years)',
        'av45': 'Beta Amyloid protein level measured by Florbetapir F18',
        'tau': 'Phosphorylated-tau deposition',
        'brain volume': 'Total Brain Matter Volume of Patient',
        'ventricular volume': 'Total Ventricular Volume of Patient',
        'moca': ' Montreal Cognitive Assessment Score',
        'brain MRI': 'brain MRI',
        'slice number': 'MRI Image slice number',
    },

    "context": "modeling the clinical and radiological phenotype of Alzheimer's Disease",
}


# ---------------------------------------------------------------------------
# Earthquake (5 nodes)
# ---------------------------------------------------------------------------
EARTHQUAKE = {
    "nodes": ['Burglary', 'Earthquake', 'Alarm', 'JohnCalls', 'Marycalls'],

    "ground_truth_edges": [
        ('Burglary', 'Alarm'),
        ('Earthquake', 'Alarm'),
        ('Alarm', 'JohnCalls'),
        ('Alarm', 'MaryCalls'),
    ],

    "descriptions": {
        'Burglary': 'burglar entering',
        'Earthquake': 'earthquake hitting',
        'Alarm': 'home alarm going off in a house',
        'JohnCalls': 'first neighbor to call to inform the alarm sound',
        'Marycalls': 'second neighbor to call to inform the alarm sound',
    },

    "context": "model factors influencing the probability of a burglary",
}


# ---------------------------------------------------------------------------
# Survey (6 nodes)
# ---------------------------------------------------------------------------
SURVEY = {
    "nodes": ['Age', 'Sex', 'Education', 'Occupation', 'Residence', 'Travel'],

    "ground_truth_edges": [
        ('Age', 'Education'),
        ('Sex', 'Education'),
        ('Education', 'Occupation'),
        ('Education', 'Residence'),
        ('Occupation', 'Travel'),
        ('Residence', 'Travel'),
    ],

    "descriptions": {
        'Age': 'Age of people using transport',
        'Sex': 'male or female',
        'Education': 'up to high school or university degree',
        'Occupation': 'employee or self-employed',
        'Residence': 'the size of the city the individual lives in, recorded as either small or big',
        'Travel': 'the means of transport favoured by the individual',
    },

    "context": "model a hypothetical survey whose aim is to investigate the usage patterns of different means of transport.",
}


# ---------------------------------------------------------------------------
# Maths (5 nodes)
# ---------------------------------------------------------------------------
MATHS = {
    "nodes": ['Mechanics', 'Vectors', 'Algebra', 'Analysis', 'Statistics'],

    "ground_truth_edges": [
        ('Vectors', 'Mechanics'),
        ('Algebra', 'Mechanics'),
        ('Algebra', 'Analysis'),
        ('Algebra', 'Statistics'),
        ('Algebra', 'Vectors'),
    ],

    "descriptions": None,

    "context": "model the relationship between different branches of mathematics and physics.",
}


# ---------------------------------------------------------------------------
# Insurance (27 nodes)
# ---------------------------------------------------------------------------
INSURANCE = {
    "nodes": [
        'GoodStudent', 'Age', 'SocioEcon', 'RiskAversion', 'VehicleYear',
        'ThisCarDam', 'RuggedAuto', 'Accident', 'MakeModel', 'DrivQuality',
        'Mileage', 'Antilock', 'DrivingSkill', 'SeniorTrain', 'ThisCarCost',
        'Theft', 'CarValue', 'HomeBase', 'AntiTheft', 'PropCost',
        'OtherCarCost', 'OtherCar', 'MedCost', 'Cushioning', 'Airbag',
        'ILiCost', 'DrivHist',
    ],

    "ground_truth_edges": [
        ('Age', 'GoodStudent'), ('Age', 'SocioEcon'), ('Age', 'RiskAversion'),
        ('Age', 'DrivingSkill'), ('Age', 'SeniorTrain'), ('Age', 'MedCost'),
        ('SocioEcon', 'GoodStudent'), ('SocioEcon', 'RiskAversion'),
        ('SocioEcon', 'VehicleYear'), ('SocioEcon', 'MakeModel'),
        ('SocioEcon', 'HomeBase'), ('SocioEcon', 'AntiTheft'),
        ('SocioEcon', 'OtherCar'),
        ('RiskAversion', 'VehicleYear'), ('RiskAversion', 'MakeModel'),
        ('RiskAversion', 'DrivQuality'), ('RiskAversion', 'SeniorTrain'),
        ('RiskAversion', 'HomeBase'), ('RiskAversion', 'AntiTheft'),
        ('RiskAversion', 'DrivHist'),
        ('VehicleYear', 'RuggedAuto'), ('VehicleYear', 'Antilock'),
        ('VehicleYear', 'CarValue'), ('VehicleYear', 'Airbag'),
        ('ThisCarDam', 'ThisCarCost'),
        ('RuggedAuto', 'ThisCarDam'), ('RuggedAuto', 'OtherCarCost'),
        ('RuggedAuto', 'Cushioning'),
        ('Accident', 'ThisCarDam'), ('Accident', 'OtherCarCost'),
        ('Accident', 'MedCost'), ('Accident', 'ILiCost'),
        ('MakeModel', 'RuggedAuto'), ('MakeModel', 'Antilock'),
        ('MakeModel', 'CarValue'), ('MakeModel', 'Airbag'),
        ('DrivQuality', 'Accident'),
        ('Mileage', 'Accident'), ('Mileage', 'CarValue'),
        ('Antilock', 'Accident'),
        ('DrivingSkill', 'DrivQuality'), ('DrivingSkill', 'DrivHist'),
        ('SeniorTrain', 'DrivingSkill'),
        ('ThisCarCost', 'PropCost'),
        ('Theft', 'ThisCarCost'),
        ('CarValue', 'ThisCarCost'), ('CarValue', 'Theft'),
        ('HomeBase', 'Theft'),
        ('AntiTheft', 'Theft'),
        ('OtherCarCost', 'PropCost'),
        ('Cushioning', 'MedCost'),
        ('Airbag', 'Cushioning'),
    ],

    "descriptions": {
        'GoodStudent': 'Car driver being a good student in driving school',
        'Age': 'Car driver age',
        'SocioEcon': 'Socio-economic status of the car driver',
        'RiskAversion': 'Car driver being risk-averse',
        'VehicleYear': 'Year in which car was bought',
        'ThisCarDam': 'Damage occurred to the car',
        'RuggedAuto': 'Ruggedness of the car',
        'Accident': 'Severity of the accident',
        'MakeModel': 'Car model',
        'DrivQuality': 'Driving quality',
        'Mileage': 'Car giving good mileage',
        'Antilock': 'Car having an anti-lock braking system',
        'DrivingSkill': 'Driving skills',
        'SeniorTrain': 'Driver receiving training from seniors',
        'ThisCarCost': 'Costs for the insured car',
        'Theft': 'Whether the car has been stolen',
        'CarValue': 'Value of the car',
        'HomeBase': 'Residential area of the car location',
        'AntiTheft': 'Car having an anti-theft system',
        'PropCost': 'Ratio of the cost for the two cars involved in the accident',
        'OtherCarCost': 'Insurance costs for the other car involved in an accident',
        'OtherCar': 'Whether any other car involved in the accident or not',
        'MedCost': 'Cost of medical treatment for the driver',
        'Cushioning': 'Car having cushioning',
        'Airbag': 'Car having an airbag',
        'ILiCost': 'Costs of inspection for insurance',
        'DrivHist': 'Driving records of the driver',
    },

    "context": "estimate the expected claim costs for a car insurance policyholder.",
}


# ---------------------------------------------------------------------------
# Sangiovese (15 nodes)
# ---------------------------------------------------------------------------
SANGIOVESE = {
    "nodes": [
        "Treatment", "SproutN", "BunchN", "GrapeW", "WoodW",
        "SPAD06", "NDVI06", "SPAD08", "NDVI08",
        "Acid", "Potass", "Brix", "pH", "Anthoc", "Polyph",
    ],

    "ground_truth_edges": [
        ('Treatment', 'SproutN'), ('Treatment', 'BunchN'),
        ('SproutN', 'BunchN'), ('Treatment', 'SPAD06'),
        ('SproutN', 'SPAD06'), ('SproutN', 'NDVI06'),
        ('SPAD06', 'NDVI06'), ('SPAD06', 'SPAD08'),
        ('NDVI06', 'SPAD08'), ('SproutN', 'NDVI08'),
        ('NDVI06', 'NDVI08'), ('SPAD08', 'NDVI08'),
        ('SproutN', 'WoodW'), ('BunchN', 'WoodW'),
        ('SPAD06', 'WoodW'), ('SPAD08', 'WoodW'),
        ('NDVI08', 'WoodW'), ('BunchN', 'Anthoc'),
        ('WoodW', 'Anthoc'), ('NDVI08', 'Anthoc'),
        ('BunchN', 'Potass'), ('SPAD06', 'Potass'),
        ('Anthoc', 'Potass'), ('Treatment', 'Brix'),
        ('Anthoc', 'Brix'), ('BunchN', 'Polyph'),
        ('NDVI06', 'Polyph'), ('NDVI08', 'Polyph'),
        ('Brix', 'Polyph'), ('Anthoc', 'Polyph'),
        ('SproutN', 'Acid'), ('BunchN', 'Acid'),
        ('SPAD06', 'Acid'), ('NDVI06', 'Acid'),
        ('NDVI08', 'Acid'), ('Brix', 'Acid'),
        ('Anthoc', 'Acid'), ('Polyph', 'Acid'),
        ('SproutN', 'pH'), ('WoodW', 'pH'),
        ('SPAD06', 'pH'), ('Acid', 'pH'),
        ('Potass', 'pH'), ('Brix', 'pH'),
        ('Anthoc', 'pH'), ('Polyph', 'pH'),
        ('SproutN', 'GrapeW'), ('BunchN', 'GrapeW'),
        ('WoodW', 'GrapeW'), ('NDVI06', 'GrapeW'),
        ('NDVI08', 'GrapeW'), ('Acid', 'GrapeW'),
        ('Brix', 'GrapeW'), ('pH', 'GrapeW'),
        ('Anthoc', 'GrapeW'),
    ],

    "descriptions": {
        "Treatment": "External decision to perform an action to improve the quality of grapes",
        "SproutN": "Mean number of grape sprouts",
        "BunchN": "Mean number of grape bunches",
        "GrapeW": "Mean weight of grapes",
        "WoodW": "Weight of grape wood",
        "SPAD06": "Soil-Plant analysis development in June",
        "NDVI06": "Normalized difference vegetation index in June",
        "SPAD08": "Soil-Plant analysis development in August",
        "NDVI08": "Normalized difference vegetation index in August",
        "Acid": "Total acidity",
        "Potass": "Potassium content",
        "Brix": "Potential alcohol",
        "pH": "Potential of hydrogen (pH)",
        "Anthoc": "Total anthocyanin content",
        "Polyph": "Total polyphenolic content",
    },

    "context": "model the causal relationships among viticultural and oenological variables of Sangiovese grapes",
}


# ---------------------------------------------------------------------------
# Neuropathic Pain (22 nodes)
# ---------------------------------------------------------------------------
NEUROPATHIC = {
    "nodes": [
        'right C7', 'right elbow trouble', 'left shoulder trouble',
        'left bend of arm problem', 'right shoulder trouble',
        'right hand problem', 'left medival elbow problems',
        'right finger trouble', 'left neck problems', 'left wrist problems',
        'left shoulder problem', 'right neck', 'right wrist problem',
        'right shoulder problem', 'discoligment injury C6-C7',
        'left hand problem', 'left C7', 'right arm band',
        'left lower arm disorders', 'neck pain', 'left finger trouble',
        'left arm',
    ],

    "ground_truth_edges": [
        ('discoligment injury C6-C7', 'left C7'),
        ('discoligment injury C6-C7', 'right C7'),
        ('left C7', 'left elbow problem'),
        ('left C7', 'left arm'),
        ('left C7', 'left neck problems'),
        ('left C7', 'neck pain'),
        ('left C7', 'left shoulder problem'),
        ('left C7', ' left finger trouble'),
        ('left C7', 'left shoulder trouble'),
        ('left C7', 'left wrist problems'),
        ('left C7', 'left medival elbow problems'),
        ('left C7', 'left hand problem'),
        ('left C7', 'left bend of arm problem'),
        ('left C7', 'left lower arm disorders'),
        ('right C7', 'neck pain'),
        ('right C7', 'left medival elbow problems'),
        ('right C7', 'right arm band'),
        ('right C7', 'right elbow trouble'),
        ('right C7', 'right hand problem'),
        ('right C7', 'right wrist problem'),
        ('right C7', 'right neck'),
        ('right C7', 'right shoulder problem'),
        ('right C7', 'right shoulder trouble'),
        ('right C7', 'right finger trouble'),
        ('right C7', 'right lower arm disorders'),
    ],

    "descriptions": None,

    "context": "for neuropathic pain diagnosis",
}


# ---------------------------------------------------------------------------
# Registry: name -> graph dict
# ---------------------------------------------------------------------------
GRAPHS = {
    "cancer": CANCER,
    "asia": ASIA,
    "child": CHILD,
    "covid": COVID,
    "alzheimers": ALZHEIMERS,
    "earthquake": EARTHQUAKE,
    "survey": SURVEY,
    "maths": MATHS,
    "insurance": INSURANCE,
    "sangiovese": SANGIOVESE,
    "neuropathic": NEUROPATHIC,
}


def get_graph(name):
    """Retrieve a graph definition by name (case-insensitive)."""
    key = name.lower()
    if key not in GRAPHS:
        available = ", ".join(sorted(GRAPHS.keys()))
        raise ValueError(f"Unknown graph '{name}'. Available: {available}")
    return GRAPHS[key]
