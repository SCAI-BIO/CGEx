# CGEx: Cypher Generating Expert

CGEx is an interactive, explainable AI system that translates **natural‑language biomedical questions** into **Cypher queries** and executes them over **Neo4j knowledge graphs**. It is designed for **COVID‑19 and neurodegenerative disease (NDD)** research, enabling users to explore evidence across multiple knowledge graphs without writing Cypher.

---

## ✨ Key Features

* **Natural Language → Cypher** using LLMs (LangChain + OpenAI)
* **Multi‑Knowledge‑Graph support** (KG selector in UI)
* **Dynamic schema extraction** from Neo4j to constrain query generation
* **Interactive Dash UI** with results, explanations, and graphs
* **Solution subgraph visualization** using Dash Cytoscape
* **Explainability (XAI)**: inspect prompts, queries, and evidence
* **Human‑in‑the‑loop feedback** (approve / disapprove generated queries)

---

## 🗂️ Project Structure

```
CGEx/
│
├── cgex.py                 # Main CGEx pipeline
├── cypher_examples.json    # Few‑shot examples
├── requirements.txt        # Python dependencies
├── README.md               # Project documentation
├── LICENSE                 # License file
├── .gitignore              # Git ignore 
```


## ⚙️ Requirements

* Python **3.9+**
* Neo4j (Aura or local)
* OpenAI API key


## 🐍 Virtual Environment (Recommended)

Create and activate a virtual environment to avoid dependency conflicts.

```bash
python -m venv .cgex         # create venv
.cgex\Scripts\activate    # activate venv on Windows
source .cgex/bin/activate # activate venv on linux
```

Install dependencies:

```bash
pip install -r requirements.txt
```


## 🧠 Knowledge Graph Setup

CGEx queries two Neo4j knowledge graphs that are maintained in separate repositories.

---

### 🔹 Knowledge Graph 1: COVID–NDD CBM Knowledge Graph

https://github.com/SCAI-BIO/CBM-Comorbidity-KG

**Setup steps:**
1. Create a Neo4j instance (Neo4j Desktop or Neo4j Aura).
2. Open Neo4j Browser.
3. Execute the Cypher import scripts provided in the repository above.
4. Verify that nodes and relationships are loaded.
5. Record the Neo4j connection details (URI, username, password).


---

### 🔹 Knowledge Graph 2: COVID–NDD Hypothesis Knowledge Graph

This knowledge graph is **already hosted** as a Neo4j Aura database and is shared:

 https://github.com/SCAI-BIO/covid-NDD-comorbidity-NLP/blob/main/src/comorbidity-hypothesis-db.py

**Important notes:**
- The Python script does **not** create or populate the knowledge graph.
- It simply opens Neo4j Browser using existing credentials.
- Users may either:
  - run the script provided in the original repository, or
  - directly access Neo4j Browser using the credentials listed there.

No additional data import is required for this graph.

## 🔐 Environment Variables

After setting up access to both knowledge graphs, add their connection details to
a `.env` file in the root of this repository:

```env
# OpenAI
OPENAI_API_KEY=your_openai_key

# Knowledge Graph 1 (CBM KG)
NEO4J_URI=bolt+s://...
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=...
NEO4J_HTTP_URI=https://...

# Knowledge Graph 2 (Hypothesis KG)
NEO4J_URI_2=bolt+s://...
NEO4J_USERNAME_2=neo4j
NEO4J_PASSWORD_2=...
NEO4J_HTTP_URI_2=https://...

```

## 🚀 Running CGEx

Start the Dash application:

```bash
python cgex.py
```

Then open your browser at:

```
http://127.0.0.1:8050
```

## 🧪 How It Works (High‑Level)

1. **User asks a question** (e.g., *"What is the relationship between COVID‑19 and Alzheimer’s disease?"*) and selects a KG.
2. CGEx **extracts the KG schema dynamically**
3. The LLM generates a **schema‑constrained Cypher query**
4. The query is **executed on the selected KG**
5. Results are returned as:
   * Raw query output
   * Natural‑language explanation
   * Interactive solution subgraph with edge evidence.
6. The user can **inspect the prompt**, **approve/disapprove** the query, and explore relationship evidence.


## 🧠 Explainability

CGEx exposes:

* The **exact prompt** sent to the LLM
* The **generated Cypher query**
* **Edge‑level evidence** (PMID, citation, source)
* Graph structure identical to Neo4j Browser (nodes + relationships)

This makes CGEx suitable for **research, clinical exploration, and hypothesis generation**.


## Contact

For any questions, suggestions, or collaborations, please contact:

Astha Anand \
Email: astha.anand@scai.fraunhofer.de 
