"""
Pairwise prompt-building functions for LLM-based causal edge orientation.

Each function constructs the `messages` list (system + assistant/user)
to be sent to the LLM. Prompt text is preserved exactly as in the
original codebase.
"""


def simple_pairwise_prompt(X, Y):
    """
    Basic pairwise prompt asking the LLM to identify the causal direction
    between two nodes X and Y, with no additional context.
    """
    question = f'''
            Which cause-and-effect relationship is more likely?

            A. changing the state of node which says {X} causally affects a change in another node which says {Y}.

            B. changing the state of node which says {Y} causally affects a change in another node which says {X}.

            C. There is no causal relationship between node {X} and {Y}.

            Make sure to first output a factually grounded reasoning for your answer. {X} and {Y} are nodes of a Causal Graph.
            The causal graph is sparse and acyclic in nature. So option C could be chosen if there is some uncertainity about causal relationship between {X} and {Y}.
            First give your reasoning and after that please make sure to provide your final answer within the tags <Answer>A/B/C</Answer>.
            It is very important that you output your final answer between the tags like <Answer>A/B/C</Answer> otherwise your response will not be processed. 
        '''

    messages = [
        {"role": "system", "content": "You are an expert in causal reasoning. You are able to accurately judge direction of edges in causal graphs."},
        {"role": "assistant", "content": f"{question}"}]

    return messages


def pairwise_with_context_prompt(X, Y, current_graph):
    """
    Pairwise prompt that provides the already-oriented part of the causal
    graph as additional context.
    """
    question = f'''
            Which cause-and-effect relationship is more likely?

            A. changing the state of node which says {X} causally affects a change in another node which says {Y}.

            B. changing the state of node which says {Y} causally affects a change in another node which says {X}.

            C. There is no causal relationship between node {X} and {Y}.

            Make sure to first output a factually grounded reasoning for your answer. {X} and {Y} are nodes of a Causal Graph.
            You can take the following part of the causal graph containing directed edges to account for causal relationship as context to help
            you arrive at the correct answer for finding the cause-and-effect relationship. Part of causal graph: {current_graph}
            The causal graph is sparse and acyclic in nature. So option C could be chosen if there is some uncertainity about causal relationship between {X} and {Y}.
            First give your reasoning and after that please make sure to provide your final answer within the tags <Answer>A/B/C</Answer>.
            It is very important that you output your final answer between the tags like <Answer>A/B/C</Answer> otherwise your response will not be processed. 
        '''
    messages = [
        {"role": "system", "content": "You are an expert in causal reasoning"},
        {"role": "assistant", "content": f"{question}"}]
    return messages


def all_directed_edges_prompt(X, Y, directed_edges):
    """
    Pairwise prompt that provides the full set of already-directed edges
    from the skeleton as context.
    """
    question = f'''
        For the nodes {X} and {Y} which form an edge in a a Causal Graph, you have to identify 
        which cause-and-effect relationship is more likely between the nodes of the edge. This will be used to rearrange the nodes in the edge 
        to create a directed edge which accounts for causal relation from one node to another in the edge.

        A. changing the state of node {X} causally affects a change in another node {Y}.

        B. changing the state of node {Y} causally affects a change in another node {X}.

        C. There is no causal relation between the nodes {X} and {Y}.

        You can also take the edges from the skeleton which have been rearranged to create a directed edge to account for causal relationship between the nodes: {directed_edges}. 
        Make sure to first output a factually grounded reasoning for your answer. First give your reasoning and after that please make sure to 
        provide your final answer within the tags <Answer>A/B/C</Answer>. 
        It is very important that you output your final answer between the tags like <Answer>A/B/C</Answer> otherwise your response will not be processed. 
        '''
    messages = [
        {"role": "system", "content": "You are an expert in causal reasoning and causal graphs. You are able to accurately judge direction of edges in causal graphs."},
        {"role": "assistant", "content": f"{question}"}]

    return messages


def markov_blanket_prompt(X, Y, X_edges, Y_edges):
    """
    Pairwise prompt that provides the Markov blanket (neighbouring directed
    edges) of nodes X and Y as context.
    """
    question = f'''
        For the following undirected edge in a Causal Graph made of nodes {X} and {Y}, you have to identify 
        which cause-and-effect relationship is more likely between the nodes of the edge. This will be used to rearrange the nodes in the edge 
        to create a directed edge which accounts for causal relation from one node to another in the edge.

        A. changing the state of node {X} causally affects a change in another node {Y}.

        B. changing the state of node {Y} causally affects a change in another node {X}.

        C. There is no causal relation between the nodes {X} and {Y}.  
        
        You can also take the other directed edges of nodes {X}: {X_edges} and {Y}: {Y_edges} of the Causal graph as context to redirect the edge to account for causal effect.
        Make sure to first output a factually grounded reasoning for your answer. First give your reasoning and after that please make sure to 
        provide your final answer within the tags <Answer>A/B/C</Answer>. 
        It is very important that you output your final answer between the tags like <Answer>A/B/C</Answer> otherwise your response will not be processed. 
        '''
    messages = [
        {"role": "system", "content": "You are an expert in causal reasoning and causal graphs. You are able to accurately judge direction of edges in causal graphs."},
        {"role": "assistant", "content": f"{question}"}]
    return messages


def remove_edges_prompt(graph):
    """
    Prompt asking the LLM to prune low-confidence edges from a dense DAG.
    """
    input_prompt = f'''For this Directed Acyclic Graph (DAG): {graph} 
        which represents a causal graph, Limit the number of edges grounded on the basis of real world knowledge. 
        The DAG is list of edges where edges are in the following format: (X1,X2) means X1 causes X2.
        
        Give priority to the edges you are most confident about in terms of causality and remove other edges. 
        Also make sure that all nodes: ['BirthAsphyxia', 'HypDistrib', 'HypoxiaInO2', 'CO2', 'ChestXray', 'Grunting', 'LVHreport', 'LowerBodyO2', 
        'RUQO2', 'CO2Report', 'XrayReport', 'Disease', 'GruntingReport', 'Age', 'LVH', 'DuctFlow', 'CardiacMixing', 
        'LungParench', 'LungFlow', 'Sick'] remain in the final graph. Show which edges should be removed and give your reasoning for the same.
        '''

    messages = [{"role": "system", "content": "You are an expert in causal reasoning"},
        {"role": "assistant", "content": f"{input_prompt}"}]

    return messages


def cot_pairwise_prompt(X, Y, context, nodes):
    """
    Chain-of-thought pairwise prompt with few-shot examples from the Cancer
    and Coronary Heart Disease graphs.
    """
    input_prompt = f''' 
        Question: For a causal graph used to model relationship of various factors and outcomes related to cancer with the following nodes: ['Pollution', 'Cancer', 'Smoker', 'Xray', 'Dyspnoea'],
            Which cause-and-effect relationship is more likely between nodes 'smoker' and 'cancer'?

            A. changing the state of node 'smoker' causally affects a change in another node 'cancer'.

            B. changing the state of node 'cancer' causally affects a change in another node 'smoker'.

            C. There is no causal relation between the nodes 'cancer' and 'smoker'.

            Make sure to first provide a grounded reasoning for your answer and then provide the answer in the following format: <Answer>A/B/C</Answer>.
            It is very important that you output the final Causal graph within the tags like <Answer>A/B/C</Answer> otherwise your answer will not be processed.
           

        Answer:  The causal effect of "smoker" directing to "cancer" is based on the strong evidence from epidemiological studies linking smoking to an increased risk of developing cancer. Smoking introduces harmful substances into the respiratory system, leading to cellular damage and mutation, which significantly raises the likelihood of cancer development in the lungs or respiratory tract, subsequently impacting the occurrence of respiratory problems like shortness of breath. Therefore answer is <Answer>A</Answer>

        Question: For a causal graph used to model relationship of various factors and outcomes related to cancer with the following nodes: ['Pollution', 'Cancer', 'Smoker', 'Xray', 'Dyspnoea'],
            Which cause-and-effect relationship is more likely between nodes 'xray' and 'dyspnoea'?

            A. changing the state of node 'xray' causally affects a change in another node 'dyspnoea'.

            B. changing the state of node 'dyspnoea' causally affects a change in another node 'xray'.

            C. There is no causal relation between the nodes 'xray' and 'dyspnoea'.

            Make sure to first provide a grounded reasoning for your answer and then provide the answer in the following format: <Answer>A/B/C</Answer>.
            It is very important that you output the final Causal graph within the tags like <Answer>A/B/C</Answer> otherwise your answer will not be processed.

        Answer: Reasoning behind the lack of causal relation between X-ray and dyspnoea is that X-ray and dyspnoea are both affects of having cancer, but they do not directly cause or affect each other. X-ray is a diagnostic test that can help detect cancer in the lungs or other organs, while dyspnoea is a symptom of cancer that involves feeling short of breath. Therefore, X-ray and dyspnoea are not causally related, but they are both associated with cancer. Therefore answer is 
        <Answer>C</Answer>

        Question: For a causal graph used to model relationship of various factors and outcomes related to cancer with the following nodes: ['Pollution', 'Cancer', 'Smoker', 'Xray', 'Dyspnoea'],
            Which cause-and-effect relationship is more likely between nodes 'xray' and 'cancer'?

            A. changing the state of node 'xray' causally affects a change in another node 'cancer'.

            B. changing the state of node 'cancer' causally affects a change in another node 'xray'.

            C. There is no causal relation between the nodes 'xray' and 'cancer'.

            Make sure to first provide a grounded reasoning for your answer and then provide the answer in the following format: <Answer>A/B/C</Answer>.
            It is very important that you output the final Causal graph within the tags like <Answer>A/B/C</Answer> otherwise your answer will not be processed.

        Answer:  The causal effect of cancer on X-ray is that X-rays are often used to diagnose or detect cancer in different parts of the body, such as the bones, lungs, breasts, or kidneys123. X-rays use low doses of radiation to create pictures of the inside of the body and show the presence, size, and location of tumors. X-rays can also help monitor the response to treatment or check for signs of recurrence. Therefore, having cancer may increase the likelihood of getting an X-ray as part of the diagnostic process or follow-up care. Therefore answer is <Answer>B</Answer>

        Question: For a causal graph used to model relationship of various factors and outcomes related to cancer with the following nodes: ['Pollution', 'Cancer', 'Smoker', 'Xray', 'Dyspnoea'],
            Which cause-and-effect relationship is more likely between nodes 'pollution' and 'cancer'?

            A. changing the state of node 'pollution' causally affects a change in another node 'cancer'.

            B. changing the state of node 'cancer' causally affects a change in another node 'pollution'.

            C. There is no causal relation between the nodes 'pollution' and 'cancer'.

           Make sure to first provide a grounded reasoning for your answer and then provide the answer in the following format: <Answer>A/B/C</Answer>.
            It is very important that you output the final Causal graph within the tags like <Answer>A/B/C</Answer> otherwise your answer will not be processed.

        Answer:  The causal effect of pollution on cancer is that air pollution contains carcinogens (cancer-causing substances) that may be absorbed into the body when inhaled and damage the DNA of cells. Another possible reasoning is that particulate matter (tiny dust-like particles) in air pollution may cause physical damage to the cells in the lungs, leading to inflammation and oxidative stress and eventually cell mutations. A third possible reasoning is that air pollution may create an inflamed environment in the lungs that encourages the proliferation of cells with existing cancer-driving mutations. These are some of the hypotheses that researchers have proposed to explain how air pollution may cause cancer, but more studies are needed to confirm them. Therefore answer is <Answer>A</Answer>

        Question: For a causal graph used to model relationship of various factors and outcomes related to cancer with the following nodes: ['Pollution', 'Cancer', 'Smoker', 'Xray', 'Dyspnoea'],
            Which cause-and-effect relationship is more likely between nodes 'pollution' and 'smoker'?

            A. changing the state of node 'pollution' causally affects a change in another node 'smoker'.

            B. changing the state of node 'smoker' causally affects a change in another node 'pollution'.

            C. There is no causal relation between the nodes 'pollution' and 'smoker'.

            Make sure to first provide a grounded reasoning for your answer and then provide the answer in the following format: <Answer>A/B/C</Answer>.
            It is very important that you output the final Causal graph within the tags like <Answer>A/B/C</Answer> otherwise your answer will not be processed.

        Answer: Reason behind the lack of causal relation between pollution and smoker is that pollution and smoking are both independent risk factors for respiratory problems, but they do not directly cause or affect each other. Pollution and smoking both contribute to air pollution, which can harm the health of people and the environment. However, pollution is mainly caused by human activities such as burning fossil fuels, deforestation, or industrial processes, while smoking is a personal choice that involves inhaling tobacco smoke. Therefore, pollution and smoker are not causally related, but they are both associated with respiratory problems. Therefore answer is <Answer>C</Answer>.

        Question: For a causal graph used for modeling factors causing Coronary Heart Diseases with the following nodes: ['Family Disease', 'Gene', 'Smoking', 'Blood Pressure', 'Coronary Heart Disease', 'Headache'],
            Which cause-and-effect relationship is more likely between nodes 'Family Disease' and 'Gene'?

            A. changing the state of node 'Family Disease' causally affects a change in another node 'Gene'.

            B. changing the state of node 'Gene' causally affects a change in another node 'Family Disease'.

            C. There is no causal relation between the nodes 'Family Disease' and 'Gene'.

            Make sure to first provide a grounded reasoning for your answer and then provide the answer in the following format: <Answer>A/B/C</Answer>.
            It is very important that you output the final Causal graph within the tags like <Answer>A/B/C</Answer> otherwise your answer will not be processed.

        Answer: Reason behind the causal effect of family disease on gene is that family disease is a term that refers to diseases or health conditions that run in the family, meaning that they are influenced by genetic factors. Gene is a term that refers to the basic unit of heredity that carries information for a specific trait or function. Family disease can affect gene by altering the type or frequency of genes that are inherited by the offspring from their parents. For example, some family diseases are caused by deterministic genes, which are genes that guarantee the development of a disease if they are present in a person's genome. Other family diseases are influenced by risk genes, which are genes that increase the likelihood of developing a disease but do not guarantee it. Therefore, family disease can causally direct to gene by changing the genetic makeup of a person or a population. Therefore answer is <Answer>A</Answer>.

        Question: For a causal graph used for modeling factors causing Coronary Heart Diseases with the following nodes: ['Family Disease', 'Gene', 'Smoking', 'Blood Pressure', 'Coronary Heart Disease', 'Headache'],
            Which cause-and-effect relationship is more likely between nodes 'Coronary Heart Disease' and 'Gene'?

            A. changing the state of node 'Coronary Heart Disease' causally affects a change in another node 'Gene'.

            B. changing the state of node 'Gene' causally affects a change in another node 'Coronary Heart Disease'.

            C. There is no causal relation between the nodes 'Coronary Heart Disease' and 'Gene'.

            Make sure to first provide a grounded reasoning for your answer and then provide the answer in the following format: <Answer>A/B/C</Answer>.
            It is very important that you output the final Causal graph within the tags like <Answer>A/B/C</Answer> otherwise your answer will not be processed.

        Answer: Possible reasoning behind the causal effect of gene on coronary heart disease is that gene is a term that refers to the basic unit of heredity that carries information for a specific trait or function. Gene can affect coronary heart disease by influencing the structure and function of the blood vessels, the metabolism and transport of lipids (fats) in the blood, the inflammation and clotting processes, or the response to environmental factors such as smoking or diet. For example, some genes code for proteins that regulate the cell cycle and growth of the cells that line the arteries, which can affect their susceptibility to damage or plaque formation. Other genes code for proteins that control the synthesis and clearance of cholesterol or other lipids, which can affect their levels and deposition in the arteries. Therefore, gene can causally direct to coronary heart disease by modifying the biological pathways that contribute to the development or progression of the disease. Therefore answer is <Answer>B</Answer>

        Question: For a causal graph used for modeling factors causing Coronary Heart Diseases with the following nodes: ['Family Disease', 'Gene', 'Smoking', 'Blood Pressure', 'Coronary Heart Disease', 'Headache'],
            Which cause-and-effect relationship is more likely between nodes 'Blood Pressure' and 'Smoking'?

            A. changing the state of node 'Blood Pressure' causally affects a change in another node 'Smoking'.

            B. changing the state of node 'Smoking' causally affects a change in another node 'Blood Pressure'.

            C. There is no causal relation between the nodes 'Blood Pressure' and 'Smoking'.

            Make sure to first provide a grounded reasoning for your answer and then provide the answer in the following format: <Answer>A/B/C</Answer>.
            It is very important that you output the final Causal graph within the tags like <Answer>A/B/C</Answer> otherwise your answer will not be processed.

        Answer: Possible reasoning behind the causal effect of smoking on blood pressure is that smoking is a habit that involves inhaling tobacco smoke, which contains nicotine and other harmful chemicals. Smoking can affect blood pressure by activating the sympathetic nervous system (SNS), which is the part of the nervous system that controls the body's response to stress or danger. When the SNS is activated, it releases hormones such as adrenaline and noradrenaline, which cause the heart to beat faster and harder, and the blood vessels to constrict. This results in a temporary increase in blood pressure, which can last for 15 to 20 minutes after each cigarette. Therefore, smoking can causally direct to blood pressure by stimulating the SNS and increasing the cardiac output and vascular resistance. Therefore answer is <Answer>B</Answer>.

        Question: For a causal graph used for modeling factors causing Coronary Heart Diseases with the following nodes: ['Family Disease', 'Gene', 'Smoking', 'Blood Pressure', 'Coronary Heart Disease', 'Headache'],
            Which cause-and-effect relationship is more likely between nodes 'Headache' and 'Smoking'?

            A. changing the state of node 'Headache' causally affects a change in another node 'Smoking'.

            B. changing the state of node 'Smoking' causally affects a change in another node 'Headache'.

            C. There is no causal relation between the nodes 'Headache' and 'Smoking'.

            Make sure to first provide a grounded reasoning for your answer and then provide the answer in the following format: <Answer>A/B/C</Answer>.
            It is very important that you output the final Causal graph within the tags like <Answer>A/B/C</Answer> otherwise your answer will not be processed.

        Answer: One possible reasoning behind the lack of causal relation between headache and smoking is that headache and smoking are both associated with various health conditions, but they do not directly cause or affect each other12. Headache is a term that refers to pain or discomfort in the head, scalp, or neck, which can have many possible causes, such as stress, dehydration, infection, injury, or medication. Smoking is a habit that involves inhaling tobacco smoke, which contains nicotine and other harmful chemicals, which can increase the risk of diseases such as cancer, heart disease, stroke, and lung disease. Therefore, headache and smoking are not causally related, but they are both linked to different health problems. Therefore the answer is <Answer>C</Answer>

        Question: For a causal graph used for modeling factors causing Coronary Heart Diseases with the following nodes: ['Family Disease', 'Gene', 'Smoking', 'Blood Pressure', 'Coronary Heart Disease', 'Headache'],
            Which cause-and-effect relationship is more likely between nodes 'Headache' and 'Smoking'?

            A. changing the state of node 'Headache' causally affects a change in another node 'Smoking'.

            B. changing the state of node 'Smoking' causally affects a change in another node 'Headache'.

            C. There is no causal relation between the nodes 'Headache' and 'Smoking'.

            Make sure to first provide a grounded reasoning for your answer and then provide the answer in the following format: <Answer>A/B/C</Answer>.
            It is very important that you output the final Causal graph within the tags like <Answer>A/B/C</Answer> otherwise your answer will not be processed.

        Answer: One possible reasoning behind the lack of causal relation between headache and smoking is that headache and smoking are both associated with various health conditions, but they do not directly cause or affect each other. Headache is a term that refers to pain or discomfort in the head, scalp, or neck, which can have many possible causes, such as stress, dehydration, infection, injury, or medication. Smoking is a habit that involves inhaling tobacco smoke, which contains nicotine and other harmful chemicals, which can increase the risk of diseases such as cancer, heart disease, stroke, and lung disease. Therefore, headache and smoking are not causally related, but they are both linked to different health problems. Therefore the answer is <Answer>C</Answer>

        Question: For a causal graph used for modeling factors causing Coronary Heart Diseases with the following nodes: ['Family Disease', 'Gene', 'Smoking', 'Blood Pressure', 'Coronary Heart Disease', 'Headache'],
            Which cause-and-effect relationship is more likely between nodes 'Coronary Heart Disease' and 'Smoking'?

            A. changing the state of node 'Smoking' causally affects a change in another node 'Coronary Heart Disease'.

            B. changing the state of node 'Coronary Heart Disease' causally affects a change in another node 'Smoking'.

            C. There is no causal relation between the nodes 'Coronary Heart Disease' and 'Smoking'.

        Make sure to first provide a grounded reasoning for your answer and then provide the answer in the following format: <Answer>A/B/C</Answer>.
        It is very important that you output the final Causal graph within the tags like <Answer>A/B/C</Answer> otherwise your answer will not be processed.

        Answer: Possible reasoning behind the causal effect of smoking on coronary heart disease is smoking damages the heart and blood vessels by raising triglycerides, lowering HDL, increasing blood clotting, and impairing blood flow to the heart. This can lead to plaque buildup, heart attacks, and death. Therefore answer is
        <Answer>A</Answer>. 
        
        
        Question: For a causal graph used for {context}. 
        Which cause-and-effect relationship is more likely between nodes {X} and {Y}?

        A. changing the state of node {X} causally affects a change in another node {Y}.

        B. changing the state of node {Y} causally affects a change in another node {X}.

        C. There is no causal relation between the nodes {X} and {Y}.

        Make sure to first provide a grounded reasoning for your answer and then provide the answer in the following format: <Answer>A/B/C</Answer>.
        It is very important that you output the final Causal graph within the tags like <Answer>A/B/C</Answer> otherwise your answer will not be processed.
        '''

    messages = [{"role": "system", "content": "You are an expert in causal reasoning. You are able to accurately judge direction of edges in causal graphs."},
        {"role": "assistant", "content": f"{input_prompt}"}]

    return messages
