"""
Triplet / subgraph prompt-building functions for LLM-based causal discovery.

Each function constructs the `messages` list (system + assistant/user)
to be sent to the LLM. Prompt text is preserved exactly as in the
original codebase.
"""


def generate_subgraph_prompt(nodes, context):
    """
    Prompt the LLM to identify causal relationships among a set of nodes
    and output a DAG as a list of directed edge tuples.
    """
    input_prompt = f''' Identify the causal relationships between the given variables and create a directed acyclic graph to {context}. 
        Make sure to give a reasoning for your answer and then output the directed graph in the form of a list of tuples, where each tuple is a directed edge. The desired output should be in the following form:
        [('A','B'), ('B','C')] where first tuple represents a directed edge from Node 'A' to Node 'B', second tuple represents a directed edge from Node 'B' to Node 'C'and so on.
         
        If a node should not form any causal relationship with other nodes, then you can add it as an isolated node of the graph by adding it seperately. For example, if 'C' should be an isolated node in a graph with nodes 'A', 'B', 'C', then the final DAG representation should be like [('A','B'), ('C')].

        It is very important that you output the final Causal graph within the tags <Answer></Answer> otherwise your answer will not be processed.
        # Example: 
        Input: ['A', 'B', 'C', 'D']
        Output: <Answer>[('A','B'),('C','D')]</Answer>
        # Question:
        Input: {nodes}
        Output:'''
    messages = [{"role": "system", "content": "You are an expert in causal reasoning. You are able to accurately judge direction of edges in causal graphs."},
        {"role": "assistant", "content": f"{input_prompt}"}]

    return messages


def generate_subgraph_with_descr_prompt(nodes, context, descr_nodes):
    """
    Prompt the LLM to identify causal relationships among a set of nodes,
    using node descriptions to improve orientation accuracy.
    """
    input_prompt = f''' Identify the causal relationships between the given variables and create a directed acyclic graph to {context}. 
        Make sure to give a reasoning for your answer and then output the directed graph in the form of a list of tuples, where each tuple is a directed edge. The desired output should be in the following form:
        [('A','B'), ('B','C')] where first tuple represents a directed edge from Node 'A' to Node 'B', second tuple represents a directed edge from Node 'B' to Node 'C'and so on.
         
        If a node should not form any causal relationship with other nodes, then you can add it as an isolated node of the graph by adding it seperately. For example, if 'C' should be an isolated node in a graph with nodes 'A', 'B', 'C', then the final DAG representation should be like [('A','B'), ('C')].
        Use the description about the node provided with the nodes in brackets to form a better decision about the causal direction orientation between the nodes.

        It is very important that you output the final Causal graph within the tags <Answer></Answer> otherwise your answer will not be processed. Do not change or modify the node names, and use the exact input node names to represent the final DAG.

        # Example: 
        Input: Nodes: ['A', 'B', 'C', 'D'];
        Description of Nodes: [description of Node A, description of Node B, description of Node C, description of Node D];
        Output: <Answer>[('A','B'),('C','D')]</Answer>
        # Question:
        Input: Nodes: {nodes};
        Description of Nodes: {descr_nodes};
        Output:'''
    messages = [{"role": "system", "content": "You are an expert in causal reasoning. You are able to accurately judge direction of edges in causal graphs."},
        {"role": "assistant", "content": f"{input_prompt}"}]

    return messages


def generate_subgraph_original_prompt(nodes):
    """
    Original prompt (no context) asking the LLM to generate a DAG from nodes.
    """
    input_prompt = f''' Identify the causal relationships between the given variables and create a directed graph to represent them. 
        Make sure to give a reasoning for your answer and then output the directed graph in the form of a list of tuples, where each tuple is a directed edge. The desired output should be in the following form:
        [('A','B'), ('B','C')] where first tuple represents a directed edge from Node 'A' to Node 'B', second tuple represents a directed edge from Node 'B' to Node 'C'and so on.
         
        If a node should not form any causal relationship with other nodes, then you can add it as an isolated node of the graph by adding it seperately. For example, if 'C' should be an isolated node in a graph with nodes 'A', 'B', 'C', then the final DAG representation should be like [('A','B'), ('C')].

        It is very important that you output the final Causal graph within the tags <Answer></Answer> otherwise your answer will not be processed.
        # Example: 
        Input: ['A', 'B', 'C', 'D']
        Output: <Answer>[('A','B'),('C','D')]</Answer>
        # Question:
        Input: {nodes}
        Output:'''
    messages = [{"role": "system", "content": "You are an expert in causal reasoning. You are able to accurately judge direction of edges in causal graphs."},
        {"role": "assistant", "content": f"{input_prompt}"}]

    return messages


def orient_edges_merge_prompt(edges):
    """
    Prompt the LLM to orient a pair of connected undirected edges.
    """
    input_prompt = f'''Identify the causal relationship between the pair of nodes ({edges[0][0]} and {edges[0][1]}), and ({edges[1][0]} and {edges[1][1]}) to orient them to form directed edges of the same causal graph used for modeling congenital heart disease. You have to identify the causal direction between both of these pairs of nodes and output the final pair of causally oriented edges in the form of list of tuples where each tuple represents a directed edge and each tuple's first element causally directs to the second element. For example if we have two pairs of nodes forming an edge like (A & B) and (B & C), if the causal relationship of the first edge is A causes B and second edge is C causes B, then the final oriented edges should be: [(A, B), (C, B)]. Also if you feel there is not a causal relationship between a given pair of nodes then do not add them in the final list of tuples. For example, if you feel (A & B) are not causally related but (B & C) are causally related, then the final oriented edges should be: [(C, B)].

        Make sure you provide a grounded reasoning for your answer, it is very important that you output the final Causal graph within the tags <Answer></Answer> otherwise your answer will not be processed.
        
        # Example:
        Input: [('A' & 'B'), ('C' & 'B')]
        Output: <Answer>[('A','B'),('B','C')]</Answer>
        # Question:
        Input: [({edges[0][0]} & {edges[0][1]}), ({edges[1][0]} & {edges[1][1]})]
        Output:
        '''

    messages = [{"role": "system", "content": "You are an expert in causal reasoning. You are able to accurately judge direction of edges in causal graphs."},
        {"role": "assistant", "content": f"{input_prompt}"}]

    return messages


def cot_tiebreaker_prompt(X, Y, nodes, context):
    """
    Chain-of-thought few-shot prompt used as a tie-breaker in the triplet
    voting merge step. Uses examples from Cancer and Coronary Heart Disease.
    """
    input_prompt = f''' 
        Question: For a causal graph used to model relationship of various factors and outcomes related to cancer with the following nodes: ['Pollution', 'Cancer', 'Smoker', 'Xray', 'Dyspnoea'],
            Which cause-and-effect relationship is more likely between nodes 'smoker' and 'cancer'?

            A. changing the state of node 'smoker' causally effects a change in another node 'cancer'.

            B. changing the state of node 'cancer' causally effects a change in another node 'smoker'.

            C. There is no causal relation between the nodes 'cancer' and 'smoker'.

            Make sure to first provide a grounded reasoning for your answer and then provide the answer in the following format: <Answer>A/B/C</Answer>.
            It is very important that you output the final Causal graph within the tags like <Answer>A/B/C</Answer> otherwise your answer will not be processed.

        Answer:  The causal effect of "smoker" directing to "cancer" is based on the strong evidence from epidemiological studies linking smoking to an increased risk of developing cancer. Smoking introduces harmful substances into the respiratory system, leading to cellular damage and mutation, which significantly raises the likelihood of cancer development in the lungs or respiratory tract, subsequently impacting the occurrence of respiratory problems like shortness of breath. Therefore answer is <Answer>A</Answer>

        Question: For a causal graph used to model relationship of various factors and outcomes related to cancer with the following nodes: ['Pollution', 'Cancer', 'Smoker', 'Xray', 'Dyspnoea'],
            Which cause-and-effect relationship is more likely between nodes 'xray' and 'dyspnoea'?

            A. changing the state of node 'xray' causally effects a change in another node 'dyspnoea'.

            B. changing the state of node 'dyspnoea' causally effects a change in anot~her node 'xray'.

            C. There is no causal relation between the nodes 'xray' and 'dyspnoea'.

            Make sure to first provide a grounded reasoning for your answer and then provide the answer in the following format: <Answer>A/B/C</Answer>.
            It is very important that you output the final Causal graph within the tags like <Answer>A/B/C</Answer> otherwise your answer will not be processed. 

        Answer: Reasoning behind the lack of causal relation between X-ray and dyspnoea is that X-ray and dyspnoea are both effects of having cancer, but they do not directly cause or affect each other. X-ray is a diagnostic test that can help detect cancer in the lungs or other organs, while dyspnoea is a symptom of cancer that involves feeling short of breath. Therefore, X-ray and dyspnoea are not causally related, but they are both associated with cancer. Therefore answer is 
        <Answer>C</Answer>

        Question: For a causal graph used to model relationship of various factors and outcomes related to cancer with the following nodes: ['Pollution', 'Cancer', 'Smoker', 'Xray', 'Dyspnoea'],
            Which cause-and-effect relationship is more likely between nodes 'xray' and 'cancer'?

            A. changing the state of node 'xray' causally effects a change in another node 'cancer'.

            B. changing the state of node 'cancer' causally effects a change in another node 'xray'.

            C. There is no causal relation between the nodes 'xray' and 'cancer'.

            Make sure to first provide a grounded reasoning for your answer and then provide the answer in the following format: <Answer>A/B/C</Answer>.
            It is very important that you output the final Causal graph within the tags like <Answer>A/B/C</Answer> otherwise your answer will not be processed.

        Answer:  The causal effect of cancer on X-ray is that X-rays are often used to diagnose or detect cancer in different parts of the body, such as the bones, lungs, breasts, or kidneys123. X-rays use low doses of radiation to create pictures of the inside of the body and show the presence, size, and location of tumors. X-rays can also help monitor the response to treatment or check for signs of recurrence. Therefore, having cancer may increase the likelihood of getting an X-ray as part of the diagnostic process or follow-up care. Therefore answer is <Answer>B</Answer>

        Question: For a causal graph used to model relationship of various factors and outcomes related to cancer with the following nodes: ['Pollution', 'Cancer', 'Smoker', 'Xray', 'Dyspnoea'],
            Which cause-and-effect relationship is more likely between nodes 'pollution' and 'cancer'?

            A. changing the state of node 'pollution' causally effects a change in another node 'cancer'.

            B. changing the state of node 'cancer' causally effects a change in another node 'pollution'.

            C. There is no causal relation between the nodes 'pollution' and 'cancer'.

           Make sure to first provide a grounded reasoning for your answer and then provide the answer in the following format: <Answer>A/B/C</Answer>.
            It is very important that you output the final Causal graph within the tags like <Answer>A/B/C</Answer> otherwise your answer will not be processed.

        Answer:  The causal effect of pollution on cancer is that air pollution contains carcinogens (cancer-causing substances) that may be absorbed into the body when inhaled and damage the DNA of cells. Another possible reasoning is that particulate matter (tiny dust-like particles) in air pollution may cause physical damage to the cells in the lungs, leading to inflammation and oxidative stress and eventually cell mutations. A third possible reasoning is that air pollution may create an inflamed environment in the lungs that encourages the proliferation of cells with existing cancer-driving mutations. These are some of the hypotheses that researchers have proposed to explain how air pollution may cause cancer, but more studies are needed to confirm them. Therefore answer is <Answer>A</Answer>

        Question: For a causal graph used to model relationship of various factors and outcomes related to cancer with the following nodes: ['Pollution', 'Cancer', 'Smoker', 'Xray', 'Dyspnoea'],
            Which cause-and-effect relationship is more likely between nodes 'pollution' and 'smoker'?

            A. changing the state of node 'pollution' causally effects a change in another node 'smoker'.

            B. changing the state of node 'smoker' causally effects a change in another node 'pollution'.

            C. There is no causal relation between the nodes 'pollution' and 'smoker'.

            Make sure to first provide a grounded reasoning for your answer and then provide the answer in the following format: <Answer>A/B/C</Answer>.
            It is very important that you output the final Causal graph within the tags like <Answer>A/B/C</Answer> otherwise your answer will not be processed.

        Answer: Reason behind the lack of causal relation between pollution and smoker is that pollution and smoking are both independent risk factors for respiratory problems, but they do not directly cause or affect each other. Pollution and smoking both contribute to air pollution, which can harm the health of people and the environment. However, pollution is mainly caused by human activities such as burning fossil fuels, deforestation, or industrial processes, while smoking is a personal choice that involves inhaling tobacco smoke. Therefore, pollution and smoker are not causally related, but they are both associated with respiratory problems. Therefore answer is <Answer>C</Answer>.

        Question: For a causal graph used for modeling factors causing Coronary Heart Diseases with the following nodes: ['Family Disease', 'Gene', 'Smoking', 'Blood Pressure', 'Coronary Heart Disease', 'Headache'],
            Which cause-and-effect relationship is more likely between nodes 'Family Disease' and 'Gene'?

            A. changing the state of node 'Family Disease' causally effects a change in another node 'Gene'.

            B. changing the state of node 'Gene' causally effects a change in another node 'Family Disease'.

            C. There is no causal relation between the nodes 'Family Disease' and 'Gene'.

            Make sure to first provide a grounded reasoning for your answer and then provide the answer in the following format: <Answer>A/B/C</Answer>.
            It is very important that you output the final Causal graph within the tags like <Answer>A/B/C</Answer> otherwise your answer will not be processed.

        Answer: Reason behind the causal effect of family disease on gene is that family disease is a term that refers to diseases or health conditions that run in the family, meaning that they are influenced by genetic factors. Gene is a term that refers to the basic unit of heredity that carries information for a specific trait or function. Family disease can affect gene by altering the type or frequency of genes that are inherited by the offspring from their parents. For example, some family diseases are caused by deterministic genes, which are genes that guarantee the development of a disease if they are present in a person's genome. Other family diseases are influenced by risk genes, which are genes that increase the likelihood of developing a disease but do not guarantee it. Therefore, family disease can causally direct to gene by changing the genetic makeup of a person or a population. Therefore answer is <Answer>A</Answer>.

        Question: For a causal graph used for modeling factors causing Coronary Heart Diseases with the following nodes: ['Family Disease', 'Gene', 'Smoking', 'Blood Pressure', 'Coronary Heart Disease', 'Headache'],
            Which cause-and-effect relationship is more likely between nodes 'Coronary Heart Disease' and 'Gene'?

            A. changing the state of node 'Coronary Heart Disease' causally effects a change in another node 'Gene'.

            B. changing the state of node 'Gene' causally effects a change in another node 'Coronary Heart Disease'.

            C. There is no causal relation between the nodes 'Coronary Heart Disease' and 'Gene'.

            Make sure to first provide a grounded reasoning for your answer and then provide the answer in the following format: <Answer>A/B/C</Answer>.
            It is very important that you output the final Causal graph within the tags like <Answer>A/B/C</Answer> otherwise your answer will not be processed.

        Answer: Possible reasoning behind the causal effect of gene on coronary heart disease is that gene is a term that refers to the basic unit of heredity that carries information for a specific trait or function. Gene can affect coronary heart disease by influencing the structure and function of the blood vessels, the metabolism and transport of lipids (fats) in the blood, the inflammation and clotting processes, or the response to environmental factors such as smoking or diet. For example, some genes code for proteins that regulate the cell cycle and growth of the cells that line the arteries, which can affect their susceptibility to damage or plaque formation. Other genes code for proteins that control the synthesis and clearance of cholesterol or other lipids, which can affect their levels and deposition in the arteries. Therefore, gene can causally direct to coronary heart disease by modifying the biological pathways that contribute to the development or progression of the disease. Therefore answer is <Answer>B</Answer>

        Question: For a causal graph used for modeling factors causing Coronary Heart Diseases with the following nodes: ['Family Disease', 'Gene', 'Smoking', 'Blood Pressure', 'Coronary Heart Disease', 'Headache'],
            Which cause-and-effect relationship is more likely between nodes 'Blood Pressure' and 'Smoking'?

            A. changing the state of node 'Blood Pressure' causally effects a change in another node 'Smoking'.

            B. changing the state of node 'Smoking' causally effects a change in another node 'Blood Pressure'.

            C. There is no causal relation between the nodes 'Blood Pressure' and 'Smoking'.

            Make sure to first provide a grounded reasoning for your answer and then provide the answer in the following format: <Answer>A/B/C</Answer>.
            It is very important that you output the final Causal graph within the tags like <Answer>A/B/C</Answer> otherwise your answer will not be processed.

        Answer: Possible reasoning behind the causal effect of smoking on blood pressure is that smoking is a habit that involves inhaling tobacco smoke, which contains nicotine and other harmful chemicals. Smoking can affect blood pressure by activating the sympathetic nervous system (SNS), which is the part of the nervous system that controls the body's response to stress or danger. When the SNS is activated, it releases hormones such as adrenaline and noradrenaline, which cause the heart to beat faster and harder, and the blood vessels to constrict. This results in a temporary increase in blood pressure, which can last for 15 to 20 minutes after each cigarette. Therefore, smoking can causally direct to blood pressure by stimulating the SNS and increasing the cardiac output and vascular resistance. Therefore answer is <Answer>B</Answer>.

        Question: For a causal graph used for modeling factors causing Coronary Heart Diseases with the following nodes: ['Family Disease', 'Gene', 'Smoking', 'Blood Pressure', 'Coronary Heart Disease', 'Headache'],
            Which cause-and-effect relationship is more likely between nodes 'Headache' and 'Smoking'?

            A. changing the state of node 'Headache' causally effects a change in another node 'Smoking'.

            B. changing the state of node 'Smoking' causally effects a change in another node 'Headache'.

            C. There is no causal relation between the nodes 'Headache' and 'Smoking'.

            Make sure to first provide a grounded reasoning for your answer and then provide the answer in the following format: <Answer>A/B/C</Answer>.
            It is very important that you output the final Causal graph within the tags like <Answer>A/B/C</Answer> otherwise your answer will not be processed.

        Answer: One possible reasoning behind the lack of causal relation between headache and smoking is that headache and smoking are both associated with various health conditions, but they do not directly cause or affect each other12. Headache is a term that refers to pain or discomfort in the head, scalp, or neck, which can have many possible causes, such as stress, dehydration, infection, injury, or medication. Smoking is a habit that involves inhaling tobacco smoke, which contains nicotine and other harmful chemicals, which can increase the risk of diseases such as cancer, heart disease, stroke, and lung disease. Therefore, headache and smoking are not causally related, but they are both linked to different health problems. Therefore the answer is <Answer>C</Answer>

        Question: For a causal graph used for modeling factors causing Coronary Heart Diseases with the following nodes: ['Family Disease', 'Gene', 'Smoking', 'Blood Pressure', 'Coronary Heart Disease', 'Headache'],
            Which cause-and-effect relationship is more likely between nodes 'Headache' and 'Smoking'?

            A. changing the state of node 'Headache' causally effects a change in another node 'Smoking'.

            B. changing the state of node 'Smoking' causally effects a change in another node 'Headache'.

            C. There is no causal relation between the nodes 'Headache' and 'Smoking'.

            Make sure to first provide a grounded reasoning for your answer and then provide the answer in the following format: <Answer>A/B/C</Answer>.
            It is very important that you output the final Causal graph within the tags like <Answer>A/B/C</Answer> otherwise your answer will not be processed.

        Answer: One possible reasoning behind the lack of causal relation between headache and smoking is that headache and smoking are both associated with various health conditions, but they do not directly cause or affect each other. Headache is a term that refers to pain or discomfort in the head, scalp, or neck, which can have many possible causes, such as stress, dehydration, infection, injury, or medication. Smoking is a habit that involves inhaling tobacco smoke, which contains nicotine and other harmful chemicals, which can increase the risk of diseases such as cancer, heart disease, stroke, and lung disease. Therefore, headache and smoking are not causally related, but they are both linked to different health problems. Therefore the answer is <Answer>C</Answer>

        Question: For a causal graph used for modeling factors causing Coronary Heart Diseases with the following nodes: ['Family Disease', 'Gene', 'Smoking', 'Blood Pressure', 'Coronary Heart Disease', 'Headache'],
            Which cause-and-effect relationship is more likely between nodes 'Coronary Heart Disease' and 'Smoking'?

            A. changing the state of node 'Smoking' causally effects a change in another node 'Coronary Heart Disease'.

            B. changing the state of node 'Coronary Heart Disease' causally effects a change in another node 'Smoking'.

            C. There is no causal relation between the nodes 'Coronary Heart Disease' and 'Smoking'.

        Make sure to first provide a grounded reasoning for your answer and then provide the answer in the following format: <Answer>A/B/C</Answer>.
        It is very important that you output the final Causal graph within the tags like <Answer>A/B/C</Answer> otherwise your answer will not be processed.

        Answer: Possible reasoning behind the causal effect of smoking on coronary heart disease is smoking damages the heart and blood vessels by raising triglycerides, lowering HDL, increasing blood clotting, and impairing blood flow to the heart. This can lead to plaque buildup, heart attacks, and death. Therefore answer is
        <Answer>A</Answer>. 
        
        
        Question: For a causal graph used {context} with the following nodes: {nodes}, 
        Which cause-and-effect relationship is more likely between nodes {X} and {Y}?

        A. changing the state of node {X} causally effects a change in another node {Y}.

        B. changing the state of node {Y} causally effects a change in another node {X}.

        C. There is no causal relation between the nodes {X} and {Y}.

        Make sure to first provide a grounded reasoning for your answer and then provide the answer in the following format: <Answer>A/B/C</Answer>.
        It is very important that you output the final Causal graph within the tags like <Answer>A/B/C</Answer> otherwise your answer will not be processed.
        '''

    messages = [{"role": "system", "content": "You are an expert in causal reasoning. You are able to accurately judge direction of edges in causal graphs."},
        {"role": "assistant", "content": f"{input_prompt}"}]

    return messages
