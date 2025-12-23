import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
import json
import re
import io
from contextlib import redirect_stdout
from langchain_openai import ChatOpenAI
from langchain_community.graphs import Neo4jGraph
from langchain.chains import GraphCypherQAChain
from langchain.prompts import PromptTemplate, FewShotPromptTemplate
from neo4j import GraphDatabase
import dotenv
import certifi
import os
import dash_cytoscape as cyto
import requests
from requests.auth import HTTPBasicAuth
import neo4j as neo4j_mod



# Load examples from the JSON file
def load_examples(file_path):
    try:
        with open(file_path, 'r') as file:
            examples_data = json.load(file)
            examples = examples_data.get('examples', [])
            updated_examples = [{"example question": ex.get("question", "No question found"), "example cypher": ex.get("cypher", "No cypher found")} for ex in examples]
            return updated_examples
    except FileNotFoundError:
        return []

# Save examples to the JSON file
def save_example(file_path, question, cypher):
    try:
        with open(file_path, 'r') as file:
            examples_data = json.load(file)
    except FileNotFoundError:
        examples_data = {"examples": []}
    
    examples_data['examples'].append({"question": question, "cypher": cypher})
    
    with open(file_path, 'w') as file:
        json.dump(examples_data, file, indent=4)

# Define the path to the examples JSON file
EXAMPLES_FILE_PATH = 'cypher_examples.json'


# Load environment variables
dotenv.load_dotenv()

# Load API key and Neo4j credentials
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# KG 1
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
NEO4J_HTTP_URI = os.getenv("NEO4J_HTTP_URI", "http://localhost:7474")


# KG 2
NEO4J_URI_2 = os.getenv("NEO4J_URI_2")
NEO4J_USERNAME_2 = os.getenv("NEO4J_USERNAME_2")
NEO4J_PASSWORD_2 = os.getenv("NEO4J_PASSWORD_2")
NEO4J_HTTP_URI_2 = os.getenv("NEO4J_HTTP_URI_2", "http://localhost:7474")

# Function to retrieve relationship details
#def extract_schema():
def extract_schema(uri, username, password):

    try:
        #driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))
        driver = GraphDatabase.driver(uri, auth=(username, password))

        with driver.session() as session:
            # Extract Nodes & Properties
            node_query = """
            CALL db.schema.nodeTypeProperties()
            YIELD nodeType, propertyName
            WITH nodeType, 
                CASE 
                    WHEN nodeType CONTAINS 'ProteinModification' THEN 
                        COLLECT(CASE WHEN propertyName IN ['name', 'amino_acid', 'position'] THEN propertyName END)
                    ELSE 
                        COLLECT(CASE WHEN propertyName = 'name' THEN propertyName END)
                END AS Properties
            RETURN nodeType AS NodeLabel, Properties
            ORDER BY NodeLabel;
            """
            node_schema = session.run(node_query).data()
        
            
            # Extract Relationships
            rel_query = """
            MATCH ()-[r]->()
            WITH DISTINCT type(r) AS relType, keys(r) AS properties
            UNWIND properties AS property
            WITH relType, COLLECT(DISTINCT property) AS uniqueProps
            WITH relType, [p IN uniqueProps WHERE p IN ["source", "citationType", "pmid", "citationRef", "evidence"]] AS filteredProps
            RETURN relType, filteredProps
            ORDER BY relType;
            """
            rel_schema = session.run(rel_query).data()
            
            
            # Directionality
            # dir_query = """
            # MATCH (a)-[r]->(b)
            # RETURN DISTINCT type(r) AS RelationshipType, labels(a)[0] AS StartNode, labels(b)[0] AS EndNode 
            # ORDER BY RelationshipType;
            # """
            # dir_schema = session.run(dir_query).data()
        
        driver.close()
        
        print("\n🔹 Extracted Nodes Schema:")
        for node in node_schema:
            print(f"  - {node['NodeLabel']} → Properties: {', '.join(node['Properties'])}")

        print("\n🔹 Extracted Relationships Schema:")
        for rel in rel_schema:
            print(f"  - {rel['relType']} → Properties: {', '.join(rel['filteredProps'])}")


        # print("\n🔹 Extracted Relationship Directionality:")
        # for dir in dir_schema:
        #     print(f"  - {dir['RelationshipType']} → Direction: {dir['StartNode']} → {dir['EndNode']}")
            
    
        return {
            "nodes": node_schema,
            "relationships": rel_schema
            # "directionality": dir_schema
        }
        

    except Exception as e:
        print(f"⚠️ Error extracting schema: {e}")
        return {"nodes": [], "relationships": [], "directionality": []}  # Fallback to avoid crashes



#def build_prompt_template(node_schema, rel_schema):
def build_prompt_template(node_schema, rel_schema, kg_name="Selected KG"):

    nodes_schema_str = "\n".join(
        [f"  - {item['NodeLabel']}\n    Properties: {', '.join(item['Properties'])}"
         for item in node_schema]
    )
    relationships_schema_str = "\n".join(
        [f"  - {item['relType']}\n    Properties: {', '.join(item['filteredProps'])}"
         for item in rel_schema]
    )

    template = f"""
You are an expert at translating user questions into Cypher Queries.
These Cypher queries are then run on a knowledge graph about COVID-19 and NDDs (Neurodegenerative diseases).

Your task is to **think step-by-step** before generating the Cypher query.

---
## **Graph in Use**
Currently using: **{kg_name}**

## **Graph Schema**
- Nodes:
{nodes_schema_str}

- Relationships:
{relationships_schema_str}

---

## **Instructions for Query Generation**

### **MANDATORY: USE THE GRAPH SCHEMA PROVIDED**
- You must refer to the graph schema when constructing Cypher queries.
- STRICTLY follow the node labels, relationship types, and property names as provided in the schemas.
- DO NOT invent node labels, properties, or relationships not present in the schema.

---

## **Node Matching Rules**
- Do NOT specify any node labels, use name filtering: 
MATCH (n) WHERE toLower(n.name) CONTAINS "covid"

- Do NOT apply conflicting conditions on the same entity. Use `OR` instead of `AND` for multiple concepts.
- If the question mentions "neurological impact", "nervous system", "neurodegeneration", etc., 
match nodes using `"neuro"`.

---

## **Relationship Matching Rules**
- First check for a direct relationship (`MATCH (a)-[r]-(b)`).
- If no direct relationship exists, use multi-hop traversal (`MATCH (a)-[*..3]-(b)`).
- Do NOT specify any relationship types. Allow general connections.
- DO NOT filter relationships using `r.name`.

---

## **ALWAYS Include `LIMIT 5` in RETURN.**

---

## **EXAMPLES (Few-Shot Reasoning + Query)**

### Example 1:
**User Question**:  
*How do cytokine levels correlate with neurodegenerative outcomes in COVID-19 patients?*

**Step-by-Step Reasoning**:
1. Identify key entities: "cytokine", "COVID", and "neurodegenerative".
2. For "cytokine", match nodes where the entity name includes "cytokine".
3. "COVID" can be matched using `toLower(d.name) CONTAINS "covid"`.
4. Neurodegenerative examples include Alzheimers, Parkinsons, or general neurodegeneration; use:
toLower(d.name) CONTAINS "alzheimer" OR toLower(d.name) CONTAINS "parkinson" OR toLower(d.name) 
CONTAINS "neuro"
5. No specific relationship types are required. Use `-[r]-`.
6. Add `LIMIT 5`.

**Cypher Query**:
```cypher
MATCH (c)-[r]-(d)-[r2]-(e)
WHERE toLower(c.name) CONTAINS "cytokine" AND 
    (toLower(d.name) CONTAINS "parkinson" OR 
    toLower(d.name) CONTAINS "alzheimer" OR 
    toLower(d.name) CONTAINS "neuro") AND
    toLower(e.name) CONTAINS "covid"
RETURN c, r, d, r2, e LIMIT 5
```

Do NOT add extra text like "Example:", "Cypher:", "---", etc., to the actual Cypher query.


Now generate a Cypher query for this question:
{{question}}

"""
    return PromptTemplate(template=template, input_variables=["question"])


# Extract schema dynamically
# try:
#     dynamic_schema = extract_schema()
# except Exception as e:
#     print(f"⚠️ Failed to extract schema from Neo4j: {e}")
#     dynamic_schema = {"nodes": [], "relationships": [], "directionality": []}

# nodes_schema = "\n".join(
#     [f"  - {item['NodeLabel']}\n    Properties: {', '.join(item['Properties'])}" 
#      for item in dynamic_schema['nodes']]
# )


# relationships_schema = "\n".join(
#     [f"  - {item['relType']}\n    Properties: {', '.join(item['filteredProps'])}"
#      for item in dynamic_schema['relationships']]
# )

# PROMPT ENGINE
# CYPHER_GENERATION_TEMPLATE = f"""
# You are an expert at translating user questions into Cypher Queries.
# These Cypher queries are then run on a knowledge graph about COVID-19 and NDDs (Neurodegenerative diseases).

# Your task is to **think step-by-step** before generating the Cypher query.

# ---

# ## **Graph Schema**
# - Nodes:
# {nodes_schema}

# - Relationships:
# {relationships_schema}

# ---

# ## **Instructions for Query Generation**

# ### **MANDATORY: USE THE GRAPH SCHEMA PROVIDED**
# - You must refer to the graph schema when constructing Cypher queries.
# - STRICTLY follow the node labels, relationship types, and property names as provided in the schemas.
# - DO NOT invent node labels, properties, or relationships not present in the schema.

# ---

# ## **Node Matching Rules**
# - Do NOT specify any node labels, use name filtering: 
# MATCH (n) WHERE toLower(n.name) CONTAINS "covid"

# - Do NOT apply conflicting conditions on the same entity. Use `OR` instead of `AND` for multiple concepts.
# - If the question mentions "neurological impact", "nervous system", "neurodegeneration", etc., 
# match nodes using `"neuro"`.

# ---

# ## **Relationship Matching Rules**
# - First check for a direct relationship (`MATCH (a)-[r]-(b)`).
# - If no direct relationship exists, use multi-hop traversal (`MATCH (a)-[*..3]-(b)`).
# - Do NOT specify any relationship types. Allow general connections.
# - DO NOT filter relationships using `r.name`.

# ---

# ## **ALWAYS Include `LIMIT 5` in RETURN.**

# ---

# ## **EXAMPLES (Few-Shot Reasoning + Query)**

# ### Example 1:
# **User Question**:  
# *How do cytokine levels correlate with neurodegenerative outcomes in COVID-19 patients?*

# **Step-by-Step Reasoning**:
# 1. Identify key entities: "cytokine", "COVID", and "neurodegenerative".
# 2. For "cytokine", match nodes where the entity name includes "cytokine".
# 3. "COVID" can be matched using `toLower(d.name) CONTAINS "covid"`.
# 4. Neurodegenerative examples include Alzheimers, Parkinsons, or general neurodegeneration; use:
# toLower(d.name) CONTAINS "alzheimer" OR toLower(d.name) CONTAINS "parkinson" OR toLower(d.name) 
# CONTAINS "neuro"
# 5. No specific relationship types are required. Use `-[r]-`.
# 6. Add `LIMIT 5`.

# **Cypher Query**:
# ```cypher
# MATCH (c)-[r]-(d)-[r2]-(e)
# WHERE toLower(c.name) CONTAINS "cytokine" AND 
#     (toLower(d.name) CONTAINS "parkinson" OR 
#     toLower(d.name) CONTAINS "alzheimer" OR 
#     toLower(d.name) CONTAINS "neuro") AND
#     toLower(e.name) CONTAINS "covid"
# RETURN c, r, d, r2, e LIMIT 5
# ```

# Do NOT add extra text like "Example:", "Cypher:", "---", etc., to the actual Cypher query.


# Now generate a Cypher query for this question:
# {{question}}

# """


# Initialize the OpenAI API
# llm = OpenAI(openai_api_key=OPENAI_API_KEY)
llm = ChatOpenAI(
    model="gpt-5",
    openai_api_key=OPENAI_API_KEY,
    model_kwargs={"response_format": {"type": "text"}}  # force text
)


# cypher_generation_prompt = PromptTemplate(
#     template=CYPHER_GENERATION_TEMPLATE,
#     input_variables=["question"],
# )

# Create a prompt template with dynamic schema details
# cypher_generation_prompt = PromptTemplate(
#     template=CYPHER_GENERATION_TEMPLATE,
#     input_variables=["question"],
# )


# Set the SSL_CERT_FILE environment variable
os.environ["SSL_CERT_FILE"] = certifi.where()

# Connect to both graphs
graph_1 = Neo4jGraph(url=NEO4J_URI, username=NEO4J_USERNAME, password=NEO4J_PASSWORD)
graph_2 = Neo4jGraph(url=NEO4J_URI_2, username=NEO4J_USERNAME_2, password=NEO4J_PASSWORD_2)


# Chains for each KG
# cypher_chain_1 = GraphCypherQAChain.from_llm(
#     llm, graph=graph_1, cypher_prompt=cypher_generation_prompt,
#     verbose=True, allow_dangerous_requests=True
# )

# cypher_chain_2 = GraphCypherQAChain.from_llm(
#     llm, graph=graph_2, cypher_prompt=cypher_generation_prompt,
#     verbose=True, allow_dangerous_requests=True
# )


def generate_cypher_fallback(llm, prompt_like, question):
    """Ask the LLM directly for a Cypher and extract it robustly."""
    # Accept either a PromptTemplate or a pre-formatted string
    if hasattr(prompt_like, "format"):
        prompt_txt = prompt_like.format(question=question)
    else:
        prompt_txt = str(prompt_like)

    prompt_txt += "\n\nReturn only the Cypher query. Do not add any explanation."
    txt = llm.invoke(prompt_txt).content  # get the string to regex


    m = re.search(r"```cypher\s*(.*?)```", txt, re.DOTALL | re.IGNORECASE)
    if not m:
        m = re.search(r"(MATCH[\s\S]*?RETURN[\s\S]*?)(?:$|\n\n|```)", txt, re.IGNORECASE)
    if m:
        cy = m.group(1).strip().replace("\\", " ")
        cy = re.sub(r"\s+", " ", cy)
        return cy
    return None



# Function to query the KG using a natural language prompt
#def query_kg(prompt, graph, cypher_chain, use_few_shot=False):
def query_kg(prompt, graph, cypher_chain, use_few_shot=False, prompt_str=None):

    try:
        inputs = {"question": prompt, "query": prompt}


        # if use_few_shot:
        #     examples = load_examples(EXAMPLES_FILE_PATH)
        #     if examples:
        #         # example_prompt_template = FewShotPromptTemplate(
        #         #     examples=examples,
        #         #     example_prompt=PromptTemplate(
        #         #         template="Example Question: {example question}\nExample Cypher: {example cypher}\n",
        #         #         input_variables=["example question", "example cypher"]
        #         #     ),
        #         #     prefix=CYPHER_GENERATION_TEMPLATE,
        #         #     suffix="Generate cypher for this Question: {question}",
        #         #     input_variables=["question"],
        #         #     example_separator="\n\n"
        #         # )
                
        #         # Rebuild schema and prompt for few-shot as well
        #         if graph == graph_1:
        #             uri, username, password = NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD
        #         else:
        #             uri, username, password = NEO4J_URI_2, NEO4J_USERNAME_2, NEO4J_PASSWORD_2

        #         schema = extract_schema(uri, username, password)
        #         base_prompt = build_prompt_template(schema["nodes"], schema["relationships"])

        #         example_prompt_template = FewShotPromptTemplate(
        #             examples=examples,
        #             example_prompt=PromptTemplate(
        #             template="Example Question: {example question}\nExample Cypher: {example cypher}\n",
        #             input_variables=["example question", "example cypher"]
        #             ),
        #             prefix=base_prompt.template,
        #             suffix="Generate cypher for this Question: {question}",
        #             input_variables=["question"],
        #             example_separator="\n\n"
        #         )


        #         cypher_chain_instance = GraphCypherQAChain.from_llm(
        #             llm,
        #             graph=graph,
        #             cypher_prompt=example_prompt_template,
        #             verbose=True,
        #             allow_dangerous_requests=True
        #         )

        #         formatted_prompt = example_prompt_template.format_prompt(question=prompt)

        #         with io.StringIO() as buf, redirect_stdout(buf):
        #             response = cypher_chain_instance(inputs)
        #             output = buf.getvalue()

        #     else:
        #         with io.StringIO() as buf, redirect_stdout(buf):
        #             response = cypher_chain(inputs)
        #             output = buf.getvalue()
        # else:
        #     with io.StringIO() as buf, redirect_stdout(buf):
        #         response = cypher_chain(inputs)
        #         output = buf.getvalue()

        # cypher_match = re.search(r'MATCH.*?RETURN.*?(?=\n|$)', output, re.DOTALL)
        # if cypher_match:
        #     generated_cypher = cypher_match.group(0).strip()
        #     generated_cypher = re.sub(r'\u001b\[0m', '', generated_cypher)  # Remove escape sequence
        #     generated_cypher = generated_cypher.replace("\n", " ")  # Replace new line with space
        #     generated_cypher = generated_cypher.replace("\\", "")  # Remove backslashes
        #     generated_cypher = re.sub(r'\s+', ' ', generated_cypher).strip()  # Remove extra spaces
        # else:
        #     generated_cypher = None
        
        if use_few_shot:
                if graph == graph_1:
                    uri, username, password = NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD
                else:
                   uri, username, password = NEO4J_URI_2, NEO4J_USERNAME_2, NEO4J_PASSWORD_2
                schema = extract_schema(uri, username, password)
                base_prompt = build_prompt_template(schema["nodes"], schema["relationships"])
                examples = load_examples(EXAMPLES_FILE_PATH)
                example_prompt_template = FewShotPromptTemplate(
                    examples=examples,
                    example_prompt=PromptTemplate(
                        template="Example Question: {example question}\nExample Cypher: {example cypher}\n",
                        input_variables=["example question", "example cypher"]
                ),
                    prefix=base_prompt.template,
                    suffix="Generate cypher for this Question: {question}",
                input_variables=["question"],
                example_separator="\n\n"
            )
                cypher_chain_instance = GraphCypherQAChain.from_llm(
                    llm, graph=graph, cypher_prompt=example_prompt_template,
                    verbose=True, allow_dangerous_requests=True, return_intermediate_steps=True
            )
                response = cypher_chain_instance.invoke(inputs)
                formatted_prompt = example_prompt_template.format_prompt(question=prompt)
        else:
                response = cypher_chain.invoke(inputs)
                #formatted_prompt = None

    # --- Robust Cypher extraction ---
    
        generated_cypher = None

        if isinstance(response, dict):
            # try direct field
            cy = response.get("cypher")
            if isinstance(cy, str) and "MATCH" in cy.upper():
                generated_cypher = cy

            # try intermediate_steps (common)
            if not generated_cypher:
                steps = response.get("intermediate_steps") or []
                # steps may be a list of dicts or list/tuple; scan from the end
                for step in reversed(steps):
                    q = None
                    if isinstance(step, dict):
                        q = step.get("query") or step.get("cypher") or step.get("tool_input")
                    elif isinstance(step, (list, tuple)) and step:
                        last = step[-1]
                        if isinstance(last, dict):
                            q = last.get("query") or last.get("cypher") or last.get("tool_input")
                    if isinstance(q, str) and "MATCH" in q.upper():
                        generated_cypher = q
                        break

            # try fenced code inside 'result'
            if not generated_cypher and isinstance(response.get("result"), str):
                m = re.search(r"```cypher\s*(.*?)```", response["result"], re.DOTALL | re.IGNORECASE)
                if not m:
                    m = re.search(r"(MATCH[\s\S]*?RETURN[\s\S]*?)(?:$|\n\n|```)", response["result"], re.IGNORECASE)
                if m:
                    generated_cypher = m.group(1).strip()

        # sanitize
        if generated_cypher:
            generated_cypher = generated_cypher.replace("\n", " ")
            generated_cypher = generated_cypher.replace("\\", "")
            generated_cypher = re.sub(r"\s+", " ", generated_cypher).strip()

    
        if not generated_cypher:
            prompt_for_chain = example_prompt_template if use_few_shot else (prompt_str or "")
            generated_cypher = generate_cypher_fallback(llm, prompt_for_chain, prompt)
            if generated_cypher:
                generated_cypher = re.sub(r"\s+", " ", generated_cypher.replace("\n", " ")).strip()

        
 
        
            # Try direct LLM fallback using the same prompt
            # decide which prompt the chain actually used


        #cypher_prompt = formatted_prompt if use_few_shot else cypher_generation_prompt.format(question=prompt)
        #cypher_prompt = str(formatted_prompt) if use_few_shot else prompt
        #cypher_prompt = str(formatted_prompt) if use_few_shot else cypher_chain.cypher_prompt.format(question=prompt)
        cypher_prompt = str(formatted_prompt) if use_few_shot else prompt_str
        

        
        if generated_cypher:
            # Dynamically choose the right credentials
            if graph == graph_1:
                uri, username, password = NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD
            else:
                uri, username, password = NEO4J_URI_2, NEO4J_USERNAME_2, NEO4J_PASSWORD_2

            results = execute_cypher(generated_cypher, uri, username, password)
            detailed_response = generate_detailed_response(results)
            return str(cypher_prompt), generated_cypher, json.dumps(results, indent=2), detailed_response

        return str(cypher_prompt), generated_cypher, None, None

    except Exception as e:
        return str(e), None, None, None

    
        

# Function to execute Cypher query on Neo4j and retrieve results
def execute_cypher(cypher_query, uri, username, password):
    driver = GraphDatabase.driver(uri, auth=(username, password))
    with driver.session() as session:
        result = session.run(cypher_query)
        return [record.data() for record in result]


# Function to generate detailed response using LLM
def generate_detailed_response(kg_results, max_tokens=550):
    response_prompt = f"""
    You are a medical expert in COVID-19 and NDD (Neurodegenerative Diseases) knowledge.
    Make sure you give complete response. It can be concise but should not be incomplete.
    Ensure that your response ends with a complete thought and does not stop abruptly.
    
    CRITICAL RULES:
    - Do NOT assume causality or direction not present.
    - If a relationship in the JSON came from an undirected pattern (e.g., '-[r]-'), 
    write "connected to" rather than directional verbs.
    - If a Relationship object includes start/end nodes, respect that direction in the wording.
    - Quote the relationship type verbatim (e.g., 'INVOLVED_IN_PATHWAY', 'INCREASES').
    
    Based on the following knowledge graph results, provide a detailed and structured response:

    {json.dumps(kg_results, indent=2)}
 
    Detailed Response:
    """
    return llm.invoke(response_prompt).content


# CSS for background image and styling
external_stylesheets = [dbc.themes.BOOTSTRAP]
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)


# Add this 👇
app.index_string = '''
<!DOCTYPE html>
<html lang="en">
<head>
    {%metas%}
    <title>CGEx: Cypher Generating Expert</title>
    {%favicon%}
    {%css%}
    <link href="https://fonts.googleapis.com/css2?family=Playfair+Display:wght@500&family=Inter:wght@400;600&display=swap" rel="stylesheet">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0-beta3/css/all.min.css" integrity="sha512-Fo3rlrZj/k7ujTTX+2n9Xl+ZlRNHl0wq35KOEqzZ19aF3b0Ql0KwUsxxW+WUcgcMEwIlWzM8qfRhYFFM2eYVFg==" crossorigin="anonymous" referrerpolicy="no-referrer" />
    <style>
        body {
            margin: 0;
            font-family: 'Inter', sans-serif;
            background: linear-gradient(to right, #e8faff, #c6ebf2);
            color: #002b36;
        }
        .topnav {
            background: rgba(0, 18, 32, 0.85);
            padding: 14px 40px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            position: sticky;
            top: 0;
            z-index: 10;
        }
        .topnav h2 {
            margin: 0;
            font-size: 1.4rem;
            font-weight: 600;
            color: #ffffff;
        }
        .topnav a {
            color: #c5e5f2;
            margin-left: 20px;
            text-decoration: none;
            font-weight: 500;
        }
        .hero {
            background-image: url('https://i0.wp.com/asiatimes.com/wp-content/uploads/2022/05/Artificial-Intelligence-AI-Quantum-Computing.jpg?fit=1200%2C794&quality=89&ssl=1');
            background-size: cover;
            background-position: center;
            height: 100vh;
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: center;
            text-align: center;
            padding: 0 20px;
            background-blend-mode: overlay;
            background-color: rgba(0, 18, 32, 0.7);
        }
        .hero h1 {
            font-family: 'Playfair Display', serif;
            font-size: 3.5rem;
            font-weight: 700;
            margin-bottom: 10px;
            color: #f2faff;
        }
        .hero p {
    font-size: 1.2rem;
    font-weight: 500;  /* Slightly bolder */
    max-width: 700px;
    margin-bottom: 30px;
    color: #f0f6fa;     /* Brighter text */
    text-shadow: 0 1px 3px rgba(0, 0, 0, 0.6);  /* Soft glow */
}

        .hero .cta-button {
            background: linear-gradient(to right, #00c1e0, #00527a);
            border: none;
            color: white;
            padding: 14px 32px;
            font-size: 1rem;
            font-weight: 600;
            border-radius: 14px;
            cursor: pointer;
            box-shadow: 0 3px 10px rgba(0, 0, 0, 0.25);
            transition: all 0.3s ease;
        }
        .hero .cta-button:hover {
            background: linear-gradient(to right, #00a6c3, #004060);
        }
        .main-content {
    max-width: 960px;
    margin: 40px auto;
    padding: 40px 30px;
    background: transparent;
    border-radius: 0;
    box-shadow: none;
}

        .section {
            margin-bottom: 40px;
        }
        .section h5 {
            font-weight: 700;
            margin-bottom: 10px;
            color: #003344;
        }
        .form-control {
            width: 100%;
            border-radius: 12px;
            padding: 12px;
            border: 1px solid #005f87;
            background-color: rgba(0, 24, 36, 0.6);
            color: #e6f7ff;
            font-size: 1rem;
            box-shadow: 0 0 5px rgba(0, 193, 224, 0);
            transition: box-shadow 0.3s ease;
        }
        .form-control:focus {
            outline: none;
            box-shadow: 0 0 10px rgba(0, 193, 224, 0.6);
        }
        .btn-primary, .btn-success, .btn-danger {
            background-color: #00a6c3 !important;
            border: none;
            color: white !important;
            padding: 10px 20px;
            border-radius: 10px;
            font-weight: 600;
            font-size: 0.95rem;
            cursor: pointer;
            margin: 5px 10px 15px 0;
            transition: background-color 0.2s ease;
        }
        .btn-primary:hover, .btn-success:hover, .btn-danger:hover {
            background-color: #0088a0 !important;
        }
        .btn-success i, .btn-danger i {
            margin-right: 6px;
        }
        pre {
            background: rgba(0, 18, 32, 0.8);
            padding: 15px;
            border-radius: 12px;
            font-size: 0.95rem;
            white-space: pre-wrap;
            color: #d0e8f0;
            border: 1px solid #004b6b;
        }
    </style>
</head>
<body>
    <div class="topnav">
        <h2>CGEx</h2>
        <div>
            <a href="#">Home</a>
            <a href="#">About</a>
            <a href="#">Use Cases</a>
            <a href="#">Contact</a>
        </div>
    </div>
    <div class="hero">
        <h1>Translate Biomedical Questions into Cypher</h1>
        <p>Built for COVID-NDD Comorbidities research. Powered by explainable AI and knowledge graphs.</p>
        <button class="cta-button" onclick="document.querySelector('.main-content').scrollIntoView({ behavior: 'smooth' });">Try It Now</button>
    </div>
    <div class="main-content">
        {%app_entry%}
    </div>
    <footer>
        {%config%}
        {%scripts%}
        {%renderer%}
    </footer>
</body>
</html>
'''


# 1. Add dropdown to select KG in layout
# 2. Update callback to include kg selection from dropdown

# 🧠 In app.layout:
app.layout = dbc.Container([
    dbc.Row([
        dbc.Col(html.H1("Cypher Generating System with XAI"), className="mb-2")
    ]),

dbc.Row([
    dbc.Col(html.Div("Enter your question:"), width=2),
    dbc.Col(dcc.Input(id='user-question', type='text', value='', style={'width': '100%'}), width=6),
    
    dbc.Col(
    dcc.Dropdown(
        id='kg-selector',
        options=[
            {'label': 'COVID–NDD CBM', 'value': 'kg1'},
            {'label': 'COVID–NDD Negin', 'value': 'kg2'}
        ],
        value='kg1',
        clearable=False,
        style={
            'color': '#000000',
            'backgroundColor': '#ffffff',
            'fontSize': '15px'
        }
    ),
    width=2,
    style={'minWidth': '180px'}
),

    
    dbc.Col(dbc.Button("Submit", id='submit-question', color='primary'), width=2)
], className="mb-4"),

    dbc.Row([
        dbc.Col(html.H5("Generated Cypher Query:"), width=12),
        dbc.Col(html.Pre(id='generated-cypher', style={'whiteSpace': 'pre-wrap', 'margin-top': '10px'}), width=12)
    ], className="mb-4"),
    dbc.Row([
        dbc.Col(html.H5("Cypher Query Results:"), width=12),
        dbc.Col(html.Pre(id='cypher-results', style={'whiteSpace': 'pre-wrap', 'margin-top': '10px'}), width=12)
    ], className="mb-4"),
    dbc.Row([
        dbc.Col(html.H5("Detailed Response:"), width=12),
        dbc.Col(html.Pre(id='detailed-response', style={'whiteSpace': 'pre-wrap', 'margin-top': '10px'}), width=12)
    ], className="mb-4"),
    
    dbc.Row([
    dbc.Col(html.H5("Solution Graph (Query Result Subgraph):"), width=12),
    dbc.Col(
        cyto.Cytoscape(
            id='solution-graph',
            layout={'name': 'cose'},            # good default force-directed layout
            style={'width': '100%', 'height': '600px', 'backgroundColor': 'white'},
            elements=[],
            stylesheet = [
    {"selector": "node", "style": {
        "label": "data(label)",
        "font-size": "12px",
        "text-wrap": "wrap",
        "text-max-width": "1000px",
        "width": 28, "height": 28,
        "background-color": "#9aa0a6"  # default grey
    }},
    # Exact labels (when real Neo4j nodes are returned)
    {"selector": 'node[labels_str *= "Protein"]',           "style": {"background-color": "#66ccff"}},  # blue
    {"selector": 'node[labels_str *= "Rna"]',               "style": {"background-color": "#4fc3f7"}},  # light blue
    {"selector": 'node[labels_str *= "GeneticFlow"]',       "style": {"background-color": "#29b6f6"}},  # mid blue
    {"selector": 'node[labels_str *= "BiologicalProcess"]', "style": {"background-color": "#ffcc80"}},  # orange
    {"selector": 'node[labels_str *= "Complex"]',           "style": {"background-color": "#ffd180"}},  # light orange
    {"selector": 'node[labels_str *= "BioConcept"]',        "style": {"background-color": "#f48fb1"}},  # pink
    {"selector": 'node[labels_str *= "Pathology"]',         "style": {"background-color": "#b39ddb"}},  # purple
    {"selector": 'node[labels_str *= "Abundance"]',         "style": {"background-color": "#a5d6a7"}},  # green
    # New KG-1 labels (add just these)
    {"selector": 'node[labels_str *= "Activity"]',           "style": {"background-color": "#4db6ac"}},  # teal
    {"selector": 'node[labels_str *= "ProteinModification"]',"style": {"background-color": "#ba68c8"}},  # purple-ish
    {"selector": 'node[labels_str *= "Reactants"]',          "style": {"background-color": "#81d4fa"}},  # light blue
    {"selector": 'node[labels_str *= "Products"]',           "style": {"background-color": "#ffe082"}},  # warm yellow
    # KG-2: only new labels
    {"selector": 'node[labels_str *= "Anatomical_Structure"]', "style": {"background-color": "#ffd54f"}},  # amber
    {"selector": 'node[labels_str *= "Pathogen"]',             "style": {"background-color": "#ef9a9a"}},  # soft red
    {"selector": 'node[labels_str *= "Pathway"]',              "style": {"background-color": "#90caf9"}},  # light blue
    {"selector": 'node[labels_str *= "Phenotype"]',            "style": {"background-color": "#d39b7d"}},  # brown (matches BioConcept vibe)
    {"selector": 'node[labels_str *= "SNP"]',                  "style": {"background-color": "#a5d6a7"}},  # green
    {"selector": 'node[labels_str *= "Symptom"]',              "style": {"background-color": "#ce93d8"}},  # lavender
    {"selector": 'node[labels_str *= "BioConceptLike"]', "style": {"background-color": "#f48fb1"}},  # match BioConcept pink
    {"selector": 'node[labels_str *= "BiologicalProcessLike"]', "style": {"background-color": "#ffcc80"}},  # orange
    {"selector": 'node[labels_str *= "Biological_Process"]', "style": {"background-color": "#ffcc80"}},  # orange like BiologicalProcess


    # Namespace buckets (kick in when we only have dict projections)
    {"selector": 'node[labels_str *= "ProteinLike"]',            "style": {"background-color": "#66ccff"}},  # HGNC
    {"selector": 'node[labels_str *= "BiologicalProcessLike"]',  "style": {"background-color": "#ffcc80"}},  # GO
    {"selector": 'node[labels_str *= "AbundanceLike"]',          "style": {"background-color": "#a5d6a7"}},  # MESH
    # DO falls back to 'Pathology' above

    {"selector": "edge", "style": {
        "label": "data(shortLabel)",
        "font-size": "10px",
        "curve-style": "bezier",
        "target-arrow-shape": "triangle",
        "text-rotation": "autorotate",
        "text-background-color": "white",
        "text-background-opacity": 0.8,
        "text-background-padding": 2
    }},
]


        ),
        width=12
    )
], className="mb-4"),
    
        dbc.Row([
        dbc.Col(html.H5("Selected Edge Evidence:"), width=12),
        dbc.Col(
            html.Pre(
                id="edge-evidence",
                style={"whiteSpace": "pre-wrap", "marginTop": "10px"}
            ),
            width=12
        )
    ], className="mb-4"),


    
    dbc.Row([
        dbc.Col(dbc.Button("Approve Query", id='approve-cypher', color='success', className="mt-2"), width=2),
        dbc.Col(dbc.Button("Disapprove Query", id='disapprove-cypher', color='danger', className="mt-2"), width=2)
    ], className="button-row"),
    dbc.Row([
        dbc.Col(html.H3("Explainability"), width=12),
        dbc.Col(html.H5("Prompt sent to the LLM:"), width=12),
        dbc.Col(html.Pre(id='cypher-prompt', style={'whiteSpace': 'pre-wrap', 'margin-top': '10px', 'text-align': 'left'}), width=12)
    ], className="mb-4")
], className="container")



# ---- helpers (place these well above the callback) ----
def format_prompt_with_examples(prompt_template, question, examples=None):
    base = prompt_template.format(question=question)
    if not examples:
        final = base
    else:
        shots = []
        for ex in examples:
            q = ex.get("example question") or ex.get("question")
            c = ex.get("example cypher") or ex.get("cypher")
            if q and c:
                shots.append(f"Example Question: {q}\nExample Cypher:\n```cypher\n{c}\n```")
        final = ("\n\n".join(shots) + "\n\n" + base) if shots else base

    # 👇 add this strict ending
    final += "\n\nReturn only the Cypher query enclosed in a ```cypher``` code block. Do not add any explanation."
    return final

CY_CODE_BLOCK = re.compile(r"```cypher\s*(.*?)```", re.DOTALL | re.IGNORECASE)
CY_FALLBACK   = re.compile(r"(MATCH[\s\S]*?RETURN[\s\S]*?)(?:$|\n\n|```)", re.IGNORECASE)

def fetch_graph_via_http(cypher_query, http_base_url, username, password, db="neo4j"):
    """
    Use Neo4j transactional HTTP endpoint to get the result as a graph
    (nodes + relationships), similar to Neo4j Browser's 'Graph' tab.
    """
    url = f"{http_base_url}/db/{db}/tx/commit"
    payload = {
        "statements": [
            {
                "statement": cypher_query,
                "resultDataContents": ["graph"]
            }
        ]
    }

    resp = requests.post(
        url,
        json=payload,
        auth=HTTPBasicAuth(username, password)
    )
    resp.raise_for_status()
    data = resp.json()

    nodes_by_id = {}
    rels_by_id = {}

    for res in data.get("results", []):
        for row in res.get("data", []):
            g = row.get("graph") or {}
            for n in g.get("nodes", []):
                nodes_by_id[n["id"]] = n
            for r in g.get("relationships", []):
                rels_by_id[r["id"]] = r

    return list(nodes_by_id.values()), list(rels_by_id.values())

def fetch_graph_via_bolt(cypher_query, uri, username, password, db="neo4j"):
    """
    Use Neo4j Python driver (Bolt) to get the result as a graph
    (nodes + relationships), similar to Neo4j Browser's 'Graph' view.
    Works with Aura (no HTTP needed).
    """
    nodes = []
    rels = []

    with GraphDatabase.driver(uri, auth=(username, password)) as driver:
        graph_obj = driver.execute_query(
            cypher_query,
            database_=db,
            result_transformer_=neo4j_mod.Result.graph,
        )
        # graph_obj has .nodes and .relationships

        for n in graph_obj.nodes:
            nid = getattr(n, "element_id", getattr(n, "id", None))
            labels = list(getattr(n, "labels", []))
            props = dict(n)
            nodes.append({
                "id": str(nid),
                "labels": labels,
                "properties": props,
            })

        for r in graph_obj.relationships:
            rid = getattr(r, "element_id", getattr(r, "id", None))
            start = getattr(r.start_node, "element_id", getattr(r.start_node, "id", None))
            end = getattr(r.end_node, "element_id", getattr(r.end_node, "id", None))
            rels.append({
                "id": str(rid),
                "type": r.type,
                "startNode": str(start),
                "endNode": str(end),
                "properties": dict(r),
            })

    return nodes, rels

def graph_to_cytoscape(nodes, rels):
    """
    Convert HTTP 'graph' result (dicts with id/labels/properties/startNode/endNode)
    into dash_cytoscape elements.
    """
    elements = []

    # Nodes
    for n in nodes:
        nid = n["id"]
        labels = n.get("labels", [])
        props = n.get("properties", {}) or {}
        name = (
            props.get("name")
            or props.get("label")
            or props.get("id")
            or nid
        )

        elements.append({
            "data": {
                "id": nid,
                "label": str(name)[:40] + ("…" if len(str(name)) > 40 else ""),
                "name_raw": str(name),
                "labels_str": ";".join(labels)
            }
        })

    # Relationships
    for r in rels:
        rid = r["id"]
        typ = r.get("type", "REL")
        src = r.get("startNode")
        tgt = r.get("endNode")
        if not (src and tgt):
            continue

        props = r.get("properties", {}) or {}

        elements.append({
            "data": {
                "id": rid,
                "source": src,
                "target": tgt,
                "label": typ,
                "shortLabel": typ[:28],

                # ✅ ADD THESE LINES
                "evidence": props.get("evidence"),
                "pmid": props.get("pmid"),
                "citationType": props.get("citationType"),
                "citationRef": props.get("citationRef"),
                "source_db": props.get("source"),
            }
        })

    return elements

def run_pipeline_direct(question, graph, uri, http_url, username, password, prompt_template, use_few_shot=False):
    examples = load_examples(EXAMPLES_FILE_PATH) if use_few_shot else None
    prompt_text = format_prompt_with_examples(prompt_template, question, examples)
    msg = llm.invoke(prompt_text)
    # Prefer plain string content
    txt = msg.content if isinstance(getattr(msg, "content", ""), str) else ""

    # If model returned list-style content (multimodal), join text parts
    if (not txt) and isinstance(msg.content, list):
        parts = []
        for p in msg.content:
            if isinstance(p, dict) and p.get("type") == "text":
                parts.append(p.get("text", ""))
        txt = " ".join(parts).strip()

    # Last resort: check additional kwargs fields some models use
    if not txt:
        ak = getattr(msg, "additional_kwargs", {}) or {}
        txt = (ak.get("content") or ak.get("message") or "").strip()

    
    print("\n--- GPT-5 raw (first 800 chars) ---\n", txt[:800], "\n-----------------------------------\n")

    m = CY_CODE_BLOCK.search(txt) or CY_FALLBACK.search(txt)
    cypher = m.group(1).strip() if m else None
    if cypher:
        cypher = re.sub(r"\s+", " ", cypher.replace("\\", " ").replace("\n", " ")).strip()
        results = execute_cypher(cypher, uri, username, password)
        detailed = generate_detailed_response(results)


        
        # 🔹 Fetch the graph via Bolt (Aura-compatible)
        nodes, rels = fetch_graph_via_bolt(cypher, uri, username, password, db="neo4j")
        elements = graph_to_cytoscape(nodes, rels)


        # If HTTP graph somehow fails but we have tabular results, fall back
        if not elements and results:
            elements = neo4j_to_cytoscape_exact(results)

        # Enrich labels for coloring (works the same as before)
        elements = enrich_labels_by_name(uri, username, password, elements)

        node_labels = [e["data"].get("labels_str", "") for e in elements if "source" not in e["data"]]
        print("[solution-graph] labels_str unique:", sorted({x for x in node_labels if x})[:12])

        return prompt_text, cypher, safe_json(results), detailed, elements
        
        #return prompt_text, cypher, json.dumps(results, indent=2), detailed
    preview = txt if len(txt) < 1500 else txt[:1500] + "\n...[truncated]"
    #return prompt_text, None, None, f"LLM returned no Cypher.\n\nRaw output preview:\n\n{preview}"
    return prompt_text, None, None, f"LLM returned no Cypher.\n\nRaw output preview:\n\n{preview}", []


from neo4j import GraphDatabase

def enrich_labels_by_name(uri, username, password, elements, max_names=200):
    """
    For nodes with empty labels_str, look up their Neo4j labels by exact name (case-insensitive)
    and fill labels_str so Cytoscape coloring works even for projected dict results.
    """
    # collect names that need enrichment
    name_to_idx = {}
    
    
    for i, el in enumerate(elements):
        d = el.get("data", {})
        if "source" in d:
            continue
        if not d.get("labels_str"):  # empty -> needs enrichment
            nm = (d.get("label") or "").strip()
        
        
        if d.get("labels_str"):     # already colored
            continue
        # use full name if present; fall back to label
        nm = (d.get("name_raw") or d.get("label") or "").strip()
        if nm:
            name_to_idx.setdefault(nm.lower(), []).append(i)

    if not name_to_idx:
        return elements  # nothing to do

    # cap to avoid huge queries
    names = list(name_to_idx.keys())[:max_names]

    driver = GraphDatabase.driver(uri, auth=(username, password))
    name_to_labels = {}
    with driver.session() as session:
        recs = session.run(
    """
    UNWIND $names AS nm
    MATCH (n)
    WHERE toLower(n.name) = nm
    WITH nm, collect(distinct labels(n)) AS labsets
    RETURN nm AS key, (CASE WHEN size(labsets) > 0 THEN labsets[0] ELSE [] END) AS labs
    """,
    names=names
)

        for r in recs:
            labs = r.get("labs") or []
            name_to_labels[r["key"]] = labs
    driver.close()

    # write labels back into elements
    for nm_lc, idxs in name_to_idx.items():
        labs = name_to_labels.get(nm_lc, [])
        if not labs:
            continue
        ls = ";".join(labs)
        for i in idxs:
            elements[i]["data"]["labels_str"] = ls

    return elements


def refetch_native_subgraph_by_names(uri, username, password, names, max_names=60):
    """
    Graph-only requery: given a list of node names, fetch a,b,r as *native* graph entities.
    Returns a dash_cytoscape elements list that mirrors Neo4j Browser.
    """
    if not names:
        return []

    # cap to avoid huge queries
    names = list({(n or "").strip() for n in names if (n or "").strip()})[:max_names]
    if not names:
        return []

    driver = GraphDatabase.driver(uri, auth=(username, password))
    with driver.session() as session:
        recs = session.run(
            """
            UNWIND $names AS nm
            MATCH (a)-[r]-(b)
            WHERE toLower(a.name) = toLower(nm) OR toLower(b.name) = toLower(nm)
            RETURN a, r, b
            """,
            names=names
        )
        records = [rec.data() for rec in recs]
    driver.close()

    return neo4j_to_cytoscape(records)


def rebuild_graph_elements_native(uri, username, password, prelim_elements):
    """
    Strict mode: rebuild elements from ONLY the edges that appeared in the prelim result.
    If nothing comes back (name collisions etc.), fall back to prelim.
    """
    pairs = pairs_from_prelim(prelim_elements)
    native_elems = refetch_native_by_pairs(uri, username, password, pairs)
    return native_elems or prelim_elements



def pairs_from_prelim(prelim_elements):
    """
    Extract the exact node-pairs (by name) that appear in the prelim Cytoscape edges.
    """
    id2name = {}
    for el in prelim_elements:
        d = el.get("data", {})
        if "source" in d:
            continue
        # prefer the raw name if present
        nm = (d.get("name_raw") or d.get("label") or "").strip()
        if nm:
            id2name[d["id"]] = nm

    pairs = []
    seen = set()
    for el in prelim_elements:
        d = el.get("data", {})
        if "source" not in d:
            continue
        a = id2name.get(d["source"])
        b = id2name.get(d["target"])
        if not a or not b:
            continue
        key = (a, b)
        if key not in seen:
            seen.add(key)
            pairs.append({"a": a, "b": b})
    return pairs


def refetch_native_by_pairs(uri, username, password, pairs):
    """
    Requery only the edges between *those* pairs (exactly what Cypher returned).
    No 1-hop neighborhood expansion.
    """
    if not pairs:
        return []
    driver = GraphDatabase.driver(uri, auth=(username, password))
    with driver.session() as s:
        recs = s.run("""
            UNWIND $pairs AS p
            MATCH (a {name:p.a})-[r]-(b {name:p.b})
            RETURN a, r, b
        """, pairs=pairs)
        records = [rec.data() for rec in recs]
    driver.close()
    return neo4j_to_cytoscape(records)


# --- exact Neo4j-Browser-style elements from the same Cypher ---
# --- Build viz exactly like Neo4j Browser, robust to UNION/LIMIT ---
import re
from neo4j import GraphDatabase

def browser_exact_elements(uri, username, password, cypher):
    """
    Re-run the same Cypher but force a Browser-style graph return by
    wrapping the original query in a subquery that returns p, then
    projecting nodes(p) and relationships(p) outside.
    Works for UNION, multiple MATCHes, and preserves per-branch LIMITs.
    Falls back if we can't find a path variable.
    """
    # Split the original query into UNION branches
    parts = re.split(r'(?is)\bUNION\b', cypher)

    # For each branch, find a path var bound via MATCH x = ( ... )
    rewritten_inside = []
    found_any_path = False
    for part in parts:
        # grab the last path var in the branch (usual case is one)
        path_matches = re.findall(r'(?is)\bMATCH\s+([A-Za-z]\w*)\s*=\s*\(', part)
        path_vars = re.findall(r'(?is)\bMATCH\s+([A-Za-z]\w*)\s*=\s*\(', part)
        
        if not path_vars:
            # keep branch as-is (it will just be ignored for viz)
            rewritten_inside.append(part.strip())
            continue

        found_any_path = True
        # keep everything up to RETURN, drop original RETURN payload, keep branch LIMIT if any
        m = re.search(r'(?is)^(.*?\b)RETURN\b.*?(?:\bLIMIT\b\s*(\d+))?\s*$', part.strip())
        if not m:
            rewritten_inside.append(part.strip())
            continue

        head = m.group(1)
        lim  = m.group(2)

        # return the path itself so outer SELECT can do nodes(p), relationships(p)
        for pv in path_vars:
            branch = f"{head}RETURN {pv} AS p"   
            if lim: branch += f" LIMIT {lim}"
            rewritten_inside.append(branch)


    if not found_any_path:
        # No path anywhere → fall back to normal conversion
        raw = execute_cypher(cypher, uri, username, password)
        return neo4j_to_cytoscape(raw)

    # Wrap all branches in one subquery, then project nodes/relationships once
    inside = " UNION ".join(rewritten_inside)
    viz_q = f"""
    CALL () {{
      {inside}
    }}
    RETURN nodes(p) AS ns, relationships(p) AS rs
    """

    print("\n[viz] Rewritten viz query:\n", viz_q, "\n")

    driver = GraphDatabase.driver(uri, auth=(username, password))
    with driver.session() as s:
        recs = s.run(viz_q).data()
    driver.close()

    # recs look like [{'ns':[Node,...],'rs':[Rel,...]}, ...]
    
    # recs look like [{'ns':[Node,...], 'rs':[Relationship,...]}, ...]
    # Build elements 1:1 from native Node/Relationship only
    def _elements_from_ns_rs(recs):
        from neo4j.graph import Node, Relationship
        import hashlib

        elements, seen_nodes, seen_edges = [], set(), set()
        id_map = {}      # element_id/id -> node cytoscape id
        name_map = {}    # lower(name) -> node cytoscape id


        def _short(s, n=40):
            s = str(s); return s if len(s) <= n else s[:n-1] + "…"

        # --- Node helpers ---
        def _add_node_native(n: Node):
            nid = str(getattr(n, "element_id", getattr(n, "id", "")))
            if nid in seen_nodes: return nid
            label = n.get("name") or next(iter(getattr(n, "labels", [])), "Node")
            elements.append({"data": {
                "id": nid,
                "label": _short(label),
                "name_raw": str(label),
                "labels_str": ";".join(list(getattr(n, "labels", [])))
            }})
            seen_nodes.add(nid)
            if nid: id_map[nid] = nid
            if label: name_map[str(label).strip().lower()] = nid
            return nid
        
        
        
        def _add_node_dict(d: dict):
            labs = d.get("labels") or []
            props = d.get("properties") or {}
            name = props.get("name") or d.get("name") or "Node"
            elemid = (
                    d.get("element_id") or d.get("elementId") or
                    d.get("id") or d.get("identity")
                )
            nid = str(elemid) if elemid is not None else hashlib.md5(str(name).encode("utf-8")).hexdigest()
            if nid in seen_nodes: return nid
            elements.append({"data": {
                "id": nid,
                "label": _short(name),
                "name_raw": str(name),
                "labels_str": ";".join([str(x) for x in labs]) if labs else ""
            }})
            seen_nodes.add(nid)
            if elemid is not None:
                id_map[str(elemid)] = nid
            if name:
                name_map[str(name).strip().lower()] = nid
            return nid
        
        

        def add_node(n):
            if isinstance(n, Node):  return _add_node_native(n)
            if isinstance(n, dict):  return _add_node_dict(n)
            # unexpected type: ignore
            return None

        # --- Relationship helpers ---
        def _add_rel_native(r: Relationship):
            rid = str(getattr(r, "element_id", r.id))
            if rid in seen_edges: return
            src = str(getattr(r.start_node, "element_id", r.start_node.id))
            tgt = str(getattr(r.end_node,   "element_id", r.end_node.id))
            # ensure endpoints exist
            add_node(r.start_node)
            add_node(r.end_node)
            elements.append({"data": {
                "id": rid, "source": src, "target": tgt,
                "label": r.type, "shortLabel": _short(r.type, 28)
            }})
            seen_edges.add(rid)

            
        def _add_rel_dict(d: dict):
            typ = d.get("type") or d.get("label") or d.get("name") or "REL"

            # try all common key variants Neo4j returns
            s = (
                d.get("start") or d.get("source") or d.get("from") or
                d.get("startNode") or d.get("start_node") or
                d.get("startNodeElementId") or d.get("startElementId") or
                d.get("startId") or d.get("start_id")
            )
            t = (
                d.get("end") or d.get("target") or d.get("to") or
                d.get("endNode") or d.get("end_node") or
                d.get("endNodeElementId") or d.get("endElementId") or
                d.get("endId") or d.get("end_id")
            )
            

            def _resolve(ep):
                # nested node dict → build and return its ID
                if isinstance(ep, dict):
                    return add_node(ep)
                # raw id (int/str) → match node we already added
                if isinstance(ep, (int, str)):
                    key = str(ep)
                    if key in id_map: return id_map[key]
                    # sometimes endpoints are names; try that too
                    nm_key = key.strip().lower()
                    if nm_key in name_map: return name_map[nm_key]
                    return None
                return None

            s_id, t_id = _resolve(s), _resolve(t)
            if not (s_id and t_id):
                return  # ⛔️ skip edges whose endpoints aren’t present

            rid = str(d.get("element_id") or d.get("elementId") or d.get("id") or d.get("identity") or f"{s_id}:{typ}:{t_id}")
            if rid in seen_edges: return
            elements.append({"data": {
                "id": rid, "source": s_id, "target": t_id,
                "label": typ, "shortLabel": _short(typ, 28)
            }})
            seen_edges.add(rid)


        def add_rel(r):
            if isinstance(r, Relationship): return _add_rel_native(r)
            if isinstance(r, dict):         return _add_rel_dict(r)
            # unexpected type: ignore
            return None

        # Build strictly from ns/rs
        for rec in recs:
            for n in (rec.get("ns") or []): add_node(n)
            for r in (rec.get("rs") or []): add_rel(r)
        return elements


    # return _strict_ elements
    return _elements_from_ns_rs(recs)


def build_viz_query_from_cypher(cypher: str):
    import re
    parts = re.split(r'(?is)\bUNION\b', cypher)
    rewritten_inside, found = [], False
    for part in parts:
        mvars = re.findall(r'(?is)\bMATCH\s+([A-Za-z]\w*)\s*=\s*\(', part)
        path_vars = re.findall(r'(?is)\bMATCH\s+([A-Za-z]\w*)\s*=\s*\(', part)
        if not path_vars:
            rewritten_inside.append(part.strip())
            continue
        found = True
        m = re.search(r'(?is)^(.*?\b)RETURN\b.*?(?:\bLIMIT\b\s*(\d+))?\s*$', part.strip())
        if not m:
            rewritten_inside.append(part.strip()); continue
            
        head, lim = m.group(1), m.group(2)
        for pv in path_vars:
            branch = f"{head}RETURN {pv} AS p"
            if lim: branch += f" LIMIT {lim}"
            rewritten_inside.append(branch)

    if not found:
        return None  # no bound path → no viz wrapper needed
    inside = " UNION ".join(rewritten_inside)
    return f"CALL () {{ {inside} }} RETURN nodes(p) AS ns, relationships(p) AS rs"


def assert_counts_match(uri, username, password, cypher, elements):
    from neo4j import GraphDatabase
    driver = GraphDatabase.driver(uri, auth=(username, password))
    with driver.session() as s:
        recs = s.run(cypher).data()
    driver.close()

    # Count native entities in raw result
    def walk(v, nodes, rels):
        from neo4j.graph import Node, Relationship, Path
        if isinstance(v, Node):
            nodes.add(getattr(v, "element_id", v.id)); return
        if isinstance(v, Relationship):
            rels.add(getattr(v, "element_id", v.id)); return
        if isinstance(v, Path):
            for n in v.nodes: nodes.add(getattr(n, "element_id", n.id))
            for r in v.relationships: rels.add(getattr(r, "element_id", r.id))
            return
        if isinstance(v, dict):
            # dict-like node
            if ("labels" in v and isinstance(v["labels"], list)) or ("properties" in v):
                nid = v.get("element_id") or v.get("id") or v.get("identity") or v.get("properties", {}).get("name")
                if nid: nodes.add(str(nid))
                return
            # dict-like relationship
            if ("type" in v) and ("start" in v or "end" in v or "source" in v or "target" in v):
                rid = v.get("element_id") or v.get("id") or f"{v.get('start') or v.get('source')}:{v.get('type')}:{v.get('end') or v.get('target')}"
                rels.add(str(rid)); return
            for x in v.values(): walk(x, nodes, rels); return
        if isinstance(v, (list, tuple, set)):
            if len(v) == 3 and all(isinstance(x, (str, dict)) for x in v):
                s, typ, t = v
                rels.add(f"{s}:{typ}:{t}"); return
            for x in v: walk(x, nodes, rels); return

    raw_nodes, raw_rels = set(), set()
    for rec in recs:
        walk(rec, raw_nodes, raw_rels)

    cy_nodes = {e["data"]["id"] for e in elements if "source" not in e["data"]}
    cy_rels  = {e["data"]["id"] for e in elements if "source" in e["data"]}

    print(f"[assert] neo4j nodes={len(raw_nodes)} vs cy nodes={len(cy_nodes)}")
    print(f"[assert] neo4j rels ={len(raw_rels)} vs cy rels ={len(cy_rels)}")
    
    extra_nodes = cy_nodes - raw_nodes
    extra_rels  = cy_rels  - raw_rels
    print(f"[assert] neo4j nodes={len(raw_nodes)} vs cy nodes={len(cy_nodes)}  extras={len(extra_nodes)}")
    print(f"[assert] neo4j rels ={len(raw_rels)} vs cy rels ={len(cy_rels)}  extras={len(extra_rels)}")
    if extra_rels:
        print("[assert] sample extra rel ids:", list(extra_rels)[:5])



# ---- end helpers ----



from neo4j.graph import Node, Relationship, Path
import hashlib

def neo4j_to_cytoscape(records):
    """
    Convert Neo4j results (list of dicts from record.data()) into dash_cytoscape 'elements'.
    Adds 'labels_str' to nodes so we can color them even if results are projected dicts.
    """

    elements, seen_nodes, seen_edges = [], set(), set()

    # ---------- helpers ----------
    def _short(s, n=32):
        s = str(s)
        return s if len(s) <= n else s[:n-1] + "…"

    def _hash_id(text):
        return "name:" + hashlib.md5(text.strip().lower().encode("utf-8")).hexdigest()

    def _name_from_dict(d):
        return str(d.get("name") or d.get("label") or d.get("bel") or d.get("id") or "Node")

    # Map namespaces to pseudo-labels (so we can color projected dicts)
    # Adjust or extend if your KG uses more namespaces (CHEBI, MONDO, HP, REACTOME, etc.)
    NS_TO_LABEL = {
        "HGNC": "ProteinLike",            # covers Protein / Rna / GeneticFlow majority
        "GO":   "BiologicalProcessLike",  # covers GO-based concepts (processes/complexes)
        "DO":   "Pathology",              # diseases
        "MESH": "AbundanceLike", # chemicals/abundance/location from MeSH
        "HP":   "BioConceptLike",   # NEW → fixes brown BioConcept color
    }

    def _labels_from_any(val):
        """
        Return a list of label-like strings for styling.
        - If we have a real Node -> return its actual labels.
        - If we have a dict projection -> try 'labels'/'label' else infer from 'namespace'.
        """
        if isinstance(val, Node):
            return list(getattr(val, "labels", []))

        if isinstance(val, dict):
            # 1) If dict already carries labels
            if "labels" in val and isinstance(val["labels"], list):
                return [str(x) for x in val["labels"] if x]

            # 2) If it has a single 'label' string (often present in your dumps)
            lab = val.get("label") or val.get("NodeLabel") or val.get("node_label")
            if isinstance(lab, str) and lab.strip():
                return [lab.strip()]

            # 3) Fall back to namespace-derived bucket
            ns = val.get("namespace")
            if isinstance(ns, str) and ns.strip():
                return [NS_TO_LABEL.get(ns.strip().upper(), ns.strip().upper())]

        return []

    def _labels_str(val):
        labs = _labels_from_any(val)
        return ";".join(labs) if labs else ""

    def add_node_from_name(name: str, labels_str: str = ""):
        nid = _hash_id(name)
        if nid in seen_nodes:
            return nid
        elements.append({"data": {
            "id": nid,
            "label": _short(name, 40),      # shown text
            "name_raw": str(name),          # <-- full name for enrichment
            "labels_str": labels_str  # <- used for color rules
        }})
        seen_nodes.add(nid)
        return nid

    def add_node(n: Node):
        nid = str(getattr(n, "element_id", n.id))
        if nid in seen_nodes:
            return nid
        label_text = n.get("name") or (list(getattr(n, "labels", []))[0] if getattr(n, "labels", None) else "Node")
        elements.append({"data": {
            "id": nid,
            "label": _short(str(label_text), 40),       # shown text
            "name_raw": str(label_text),                # <-- full name for enrichment
            "labels_str": ";".join(list(getattr(n, "labels", [])))  # real Neo4j labels if available
        }})
        seen_nodes.add(nid)
        return nid

    def label_from_rel_like(val):
        if isinstance(val, Relationship):
            return val.type
        if isinstance(val, str):
            return val
        if isinstance(val, tuple):
            for x in val:
                if isinstance(x, str) and (x.isupper() or len(x) <= 24):
                    return x
            for x in val:
                if isinstance(x, str):
                    return x
            return "REL"
        if isinstance(val, dict):
            return val.get("type") or val.get("label") or val.get("name") or "REL"
        return "REL"

    def add_edge_by_ids(src_id: str, tgt_id: str, label: str):
        rid = f"{src_id}:{label}:{tgt_id}"
        if rid in seen_edges:
            return
        elements.append({"data": {
            "id": rid,
            "source": src_id,
            "target": tgt_id,
            "label": label,
            "shortLabel": _short(label, 28)
        }})
        seen_edges.add(rid)
        

    def add_edge(r: Relationship):
        src_id = add_node(r.start_node)
        tgt_id = add_node(r.end_node)
        rid = str(getattr(r, "element_id", r.id))
        if rid in seen_edges:
            return
        elements.append({"data": {
        "id": rid,
        "source": str(getattr(r.start_node, "element_id", r.start_node.id)),
        "target": str(getattr(r.end_node, "element_id", r.end_node.id)),
        "label": r.type,
        "shortLabel": _short(r.type, 28)
        }})
        seen_edges.add(rid)
        

    def is_node_like(val):
        if isinstance(val, Node):
            return True
        if isinstance(val, dict):
            return any(k in val for k in ("name", "label", "bel", "labels", "NodeLabel", "id", "namespace", "node_label"))
        return False

    def node_id_from_any(val):
        if isinstance(val, Node):
            return add_node(val)
        if isinstance(val, dict):
            return add_node_from_name(_name_from_dict(val), _labels_str(val))
        return None

    def try_infer_triplets(seq):
        # infer edges from (node-like, rel-like, node-like)
        for i in range(len(seq) - 2):
            v1, v2, v3 = seq[i], seq[i+1], seq[i+2]
            if is_node_like(v1) and not is_node_like(v2) and is_node_like(v3):
                src = node_id_from_any(v1)
                tgt = node_id_from_any(v3)
                lbl = label_from_rel_like(v2)
                if src and tgt:
                    add_edge_by_ids(src, tgt, lbl)

    def walk(val):
        if isinstance(val, Node):
            add_node(val); return
        if isinstance(val, Relationship):
            add_edge(val); return
        if isinstance(val, Path):
            for n in val.nodes: add_node(n)
            for r in val.relationships: add_edge(r)
            return

        if isinstance(val, dict):
            items = list(val.items())
            values_in_order = [v for _, v in items]
            for v in values_in_order:
                if is_node_like(v): node_id_from_any(v)
            try_infer_triplets(values_in_order)
            for v in values_in_order: walk(v)
            return

        if isinstance(val, (list, tuple, set)):
            if len(val) == 3 and is_node_like(val[0]) and is_node_like(val[2]):
                src = node_id_from_any(val[0])
                tgt = node_id_from_any(val[2])
                lbl = label_from_rel_like(val[1])
                if src and tgt:
                    add_edge_by_ids(src, tgt, lbl)
            for v in val: walk(v)
            return
        # primitives -> ignore

    for rec in records:
        walk(rec)

    return elements


# Draw EXACTLY what the query returned: Nodes, Relationships, Paths.
from neo4j.graph import Node, Relationship, Path
import hashlib

def neo4j_to_cytoscape_exact(records):
    elements, seen_nodes, seen_edges = [], set(), set()
    id_map = {}      # maps element_id/hash -> cytoscape node id
    name_map = {}    # maps lowercased name -> cytoscape node id


    def _short(s, n=32):
        s = str(s); return s if len(s) <= n else s[:n-1] + "…"
    def _nid_from_node(n: Node):
        nid = str(getattr(n, "element_id", n.id))
        if nid in seen_nodes: return nid
        label_text = n.get("name") or (list(getattr(n, "labels", []))[0] if getattr(n, "labels", None) else "Node")
        elements.append({"data": {
            "id": nid,
            "label": _short(str(label_text), 40),
            "name_raw": str(label_text),
            "labels_str": ";".join(list(getattr(n, "labels", [])))
        }})
        seen_nodes.add(nid)
        id_map[nid] = nid
        name_map[str(label_text).strip().lower()] = nid
        return nid
    

    def _add_rel(r: Relationship):
        s = _nid_from_node(r.start_node)
        t = _nid_from_node(r.end_node)
        # AFTER (1:1 with Neo4j results)
        rid = str(getattr(r, "element_id", r.id))
        if rid in seen_edges: 
            return
        elements.append({"data": {
                "id": rid,
                "source": str(getattr(r.start_node, "element_id", r.start_node.id)),
                "target": str(getattr(r.end_node, "element_id", r.end_node.id)),
                "label": r.type,
                "shortLabel": _short(r.type, 28)
                    }})
        seen_edges.add(rid)
        
    
    def _nid_from_node_dict(d: dict):
        # Accept dicts that look like nodes: have labels and properties (common REST-like shape)
        labs = d.get("labels") or []
        props = d.get("properties") or {}
        name = props.get("name") or d.get("name") or "Node"
        # Use element_id if present; else fall back to hash of name (still stable enough for viz)
        elemid = d.get("element_id") or d.get("id")
        nid = str(elemid) if elemid else hashlib.md5(str(name).encode("utf-8")).hexdigest()
        if nid in seen_nodes: 
            return nid
        elements.append({"data": {
            "id": nid,
            "label": _short(str(name), 40),
            "name_raw": str(name),
            "labels_str": ";".join([str(x) for x in labs]) if labs else ""
        }})
        seen_nodes.add(nid)
        if elemid: id_map[str(elemid)] = nid
        name_map[str(name).strip().lower()] = nid
        return nid


    def _add_rel_from_dict(d: dict):
        typ = d.get("type") or d.get("label") or d.get("name") or "REL"
        s = d.get("start") or d.get("source")
        t = d.get("end")   or d.get("target")

        def resolve_endpoint(ep):
            # If nested node-dict → build it and return its cytoscape id
            if isinstance(ep, dict):
                return _nid_from_node_dict(ep)
            # If it’s an element_id string → only use if we already have that node
            if isinstance(ep, str) and ep in id_map:
                return id_map[ep]
            # If it looks like a name → try name_map
            if isinstance(ep, str) and ep.strip().lower() in name_map:
                return name_map[ep.strip().lower()]
            return None

        s_id = resolve_endpoint(s)
        t_id = resolve_endpoint(t)
        if not (s_id and t_id):
            return  # ❗ don’t fabricate endpoints

        rid = str(d.get("element_id") or d.get("id") or f"{s_id}:{typ}:{t_id}")
        if rid in seen_edges: return
        elements.append({"data": {"id": rid, "source": s_id, "target": t_id,
                                "label": typ, "shortLabel": _short(typ, 28)}})
        seen_edges.add(rid)



    # Walk ONLY Node/Relationship/Path. Do NOT infer from dict/tuple/list shapes.
    def walk(v):
        if isinstance(v, Node): _nid_from_node(v); return
        if isinstance(v, Relationship): _add_rel(v); return
        if isinstance(v, Path):
            for n in v.nodes: _nid_from_node(n)
            for r in v.relationships: _add_rel(r)
            return
        if isinstance(v, dict):
            # If dict looks like a node/rel, handle directly
            if ("labels" in v and (isinstance(v["labels"], list))) or ("properties" in v):
                _nid_from_node_dict(v); return
            if ("type" in v and ("start" in v or "end" in v or "source" in v or "target" in v)):
                _add_rel_from_dict(v); return
            # else: recurse into values
            for x in v.values(): walk(x)
            return
        if isinstance(v, (list, tuple, set)):
            # Some drivers return rel as ("start","TYPE","end")
            if len(v) == 3:
                s, typ, t = v
                def resolve_seq_end(ep):
                    if isinstance(ep, dict): return _nid_from_node_dict(ep)
                    if isinstance(ep, str) and ep in id_map: return id_map[ep]
                    if isinstance(ep, str) and ep.strip().lower() in name_map: return name_map[ep.strip().lower()]
                    return None
                
                s_id = resolve_seq_end(s)
                t_id = resolve_seq_end(t)
                if s_id and t_id:
                    _add_rel_from_dict({"type": str(typ) if not isinstance(typ, dict) else (typ.get("type") or "REL"),
                                "start": s_id, "end": t_id})
                    return
            for x in v: walk(x)
            return
    # primitives: ignore

    for rec in records:
        walk(rec)
    return elements


def safe_json(obj):   
    from neo4j.graph import Node, Relationship
    import json
    def coerce(o):
        if isinstance(o, Node):
            return {"node_id": o.id, "labels": list(o.labels), **dict(o)}
        if isinstance(o, Relationship):
            return {"rel_id": o.id, "type": o.type,
                    "start": o.start_node.id, "end": o.end_node.id, **dict(o)}
        if isinstance(o, (list, tuple)):
            return [coerce(x) for x in o]
        if isinstance(o, dict):
            return {k: coerce(v) for k, v in o.items()}
        return o
    try:
        return json.dumps(obj, indent=2)
    except TypeError:
        return json.dumps(coerce(obj), indent=2)



# 🧠 Update callback to take dropdown input
# 🧠 Update callback to take dropdown input
@app.callback(
    Output('generated-cypher', 'children'),
    Output('detailed-response', 'children'),
    Output('cypher-results', 'children'),
    Output('cypher-prompt', 'children'),
    Output('solution-graph', 'elements'),      # 👈 NEW
    Input('submit-question', 'n_clicks'),
    Input('approve-cypher', 'n_clicks'),
    Input('disapprove-cypher', 'n_clicks'),
    State('user-question', 'value'),
    State('kg-selector', 'value'),
    State('generated-cypher', 'children'),
    State('cypher-prompt', 'children'),
    prevent_initial_call=True
)

def update_output(submit_clicks, approve_clicks, disapprove_clicks, question, selected_kg, generated_cypher, cypher_prompt):
    ctx = dash.callback_context

    if not ctx.triggered:
        #return '', '', '', ''
        return '', '', '', '', dash.no_update


    button_id = ctx.triggered[0]['prop_id'].split('.')[0]

    # 🔀 Select correct graph + chain based on dropdown
    # if selected_kg == 'kg1':
    #     graph, chain = graph_1, cypher_chain_1
    # else:
    #     graph, chain = graph_2, cypher_chain_2
    
    if selected_kg == 'kg1':
        uri, username, password = NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD
        http_url = NEO4J_HTTP_URI
        graph = graph_1
    else:
        uri, username, password = NEO4J_URI_2, NEO4J_USERNAME_2, NEO4J_PASSWORD_2
        http_url = NEO4J_HTTP_URI_2
        graph = graph_2


    # 💡 Extract the dynamic schema and build the prompt
    #schema = extract_schema(uri, username, password)
    #prompt_template = build_prompt_template(schema["nodes"], schema["relationships"])
    
    schema = extract_schema(uri, username, password)

    if selected_kg == 'kg1':
        kg_name = "COVID–NDD CBM KG"
    else:
        kg_name = "COVID–NDD Negin KG"

    prompt_template = build_prompt_template(schema["nodes"], schema["relationships"], kg_name=kg_name)


    # 💡 Create a Cypher QA chain with the dynamic prompt
    # cypher_chain = GraphCypherQAChain.from_llm(
    #     llm, graph=graph, cypher_prompt=prompt_template,
    #     verbose=True, allow_dangerous_requests=True,
    #     return_intermediate_steps=True
    #     )


    if button_id == 'submit-question' and question:
        #cypher_prompt, generated_cypher, cypher_results, detailed_response = query_kg(question, graph, chain)
        #cypher_prompt, generated_cypher, cypher_results, detailed_response = query_kg(question, graph, cypher_chain)
        # formatted_prompt_str = prompt_template.format(question=question)
        # cypher_prompt, generated_cypher, cypher_results, detailed_response = query_kg(question, graph, cypher_chain, prompt_str=formatted_prompt_str)
        cypher_prompt, generated_cypher, cypher_results, detailed_response, elements = run_pipeline_direct(
         question, graph, uri, http_url, username, password, prompt_template, use_few_shot=False
)

        return generated_cypher, detailed_response, cypher_results, cypher_prompt, elements

    elif button_id == 'approve-cypher' and generated_cypher:
        save_example(EXAMPLES_FILE_PATH, question, generated_cypher)
        return generated_cypher, 'Cypher query approved and saved.', '', cypher_prompt, dash.no_update


    elif button_id == 'disapprove-cypher':
        examples = load_examples(EXAMPLES_FILE_PATH)
        if examples:
            #cypher_prompt, generated_cypher, cypher_results, detailed_response = query_kg(question, graph, chain, use_few_shot=True)
            #cypher_prompt, generated_cypher, cypher_results, detailed_response = query_kg(question, graph, cypher_chain, use_few_shot=True)
            # formatted_prompt_str = prompt_template.format(question=question)
            # cypher_prompt, generated_cypher, cypher_results, detailed_response = query_kg(question, graph, cypher_chain, use_few_shot=True, prompt_str=formatted_prompt_str)
            cypher_prompt, generated_cypher, cypher_results, detailed_response, elements = run_pipeline_direct(
    question, graph, uri, http_url, username, password, prompt_template, use_few_shot=True
)
            
    
            return generated_cypher, detailed_response, cypher_results, cypher_prompt, elements
        else:
            #return '', 'Cypher query disapproved. No examples available for few-shot learning.', '', ''
            return '', 'Cypher query disapproved. No examples available for few-shot learning.', '', '', dash.no_update

    #return '', '', '', ''
    return '', '', '', '', []


@app.callback(
    Output("edge-evidence", "children"),
    Input("solution-graph", "tapEdgeData")
)
def show_edge_evidence(edge_data):
    if not edge_data:
        return "Click an edge to view its evidence."

    ev = edge_data.get("evidence") or "<no evidence available>"
    pmid = edge_data.get("pmid")
    ctype = edge_data.get("citationType")
    src = edge_data.get("source_db")

    lines = [f"Evidence:\n{ev}"]
    if pmid:
        lines.append(f"\nPMID: {pmid}")
    if src or ctype:
        lines.append(f"Source: {src or ''} {ctype or ''}".strip())

    return "\n".join(lines)



if __name__ == '__main__':
    #app.run_server(debug=True)
    app.run(debug=True)
    #app.run_server(debug=False)
    
    

