import os
from datetime import datetime
from rdflib import Graph, Namespace, URIRef, Literal, RDF, OWL, XSD

# CONFIG

AQ = Namespace("http://aqi-prediction.org/ontology#")
# Save in the same directory as the script
OUTPUT_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "final_aqi_knowledge_graph.ttl")

# INITIALIZE GRAPH + ONTOLOGY
# =========================================================

def init_graph():
    g = Graph()
    g.bind("aq", AQ)

    # Core classes
    g.add((AQ.EnvironmentalFeature, RDF.type, OWL.Class))
    g.add((AQ.Prediction, RDF.type, OWL.Class))

    # Core properties
    properties = [
        AQ.hasObservedValue,
        AQ.hasPredictedValue,
        AQ.hasFeatureInput,
        AQ.hasLikelySource,
        AQ.hasHealthRisk,
        AQ.mitigationAdvice,
        AQ.affectedGroup,
        AQ.regulatoryStatus,
        AQ.dataQualityFlag,
        AQ.chemicalSynergy,
        AQ.hasDynamicWarning,
        AQ.hasContextualInsight,
        AQ.governmentRecommendation  # <-- NEW: Gov Solution Schema
    ]

    for prop in properties:
        g.add((prop, RDF.type, RDF.Property))

    return g

#STATIC EXPERT KNOWLEDGE (ALWAYS PRESENT)
# =========================================================

def add_expert_knowledge(g):
    knowledge = {
        "PM2.5": {
            "source": "Vehicle exhaust, construction dust",
            "risk": "Penetrates deep into lungs, asthma aggravation",
            "mitigation": "Use air purifiers, wear N95 masks, restrict outdoor exercise",
            "vulnerable": "Children, Elderly, Asthmatics",
            "gov_solution": "Invoke GRAP Stage protocols, halt non-essential construction, deploy anti-smog guns."
        },
        "PM10": {
            "source": "Dust storms, agriculture",
            "risk": "Lung irritation, coughing",
            "mitigation": "Wet sweeping of roads, cover construction sites",
            "vulnerable": "Construction workers, people with bronchitis",
            "gov_solution": "Mandate mechanized sweeping of roads, water sprinkling on unpaved roads."
        },
        "NO2": {
            "source": "Burning of fossil fuels (cars)",
            "risk": "Inflammation of airways, reduced lung function",
            "mitigation": "Avoid traffic-heavy routes, carpool, use electric vehicles",
            "vulnerable": "Children with asthma",
            "gov_solution": "Divert heavy traffic, enforce strict PUC checks, encourage public transport."
        },
        "Ozone": {
            "source": "Chemical reaction in sunlight",
            "risk": "Chest pain, throat irritation",
            "mitigation": "Refuel cars in the evening, limit daytime driving",
            "vulnerable": "Outdoor workers, active children",
            "gov_solution": "Restrict industrial VOC emissions during peak daylight hours."
        },
        "CO": {
            "source": "Incomplete combustion",
            "risk": "Reduces oxygen delivery to body's organs",
            "mitigation": "Ensure proper ventilation indoors, check car exhaust",
            "vulnerable": "People with heart disease",
            "gov_solution": "Regulate idling of vehicles at traffic intersections, enforce emission limits."
        }
    }

    for name, info in knowledge.items():
        feat_uri = URIRef(f"{AQ}Feature_{name}")
        g.add((feat_uri, RDF.type, AQ.EnvironmentalFeature))
        g.add((feat_uri, AQ.hasLikelySource, Literal(info["source"])))
        g.add((feat_uri, AQ.hasHealthRisk, Literal(info["risk"])))
        g.add((feat_uri, AQ.mitigationAdvice, Literal(info["mitigation"])))
        g.add((feat_uri, AQ.affectedGroup, Literal(info["vulnerable"])))
        g.add((feat_uri, AQ.governmentRecommendation, Literal(info["gov_solution"]))) # <-- NEW

    print("✅ Expert knowledge & Gov Solutions injected into RDF graph.")

# PIPELINE INTEGRATION (CALL THIS FROM aqi_pipeline.py)
# =========================================================

def update_24hr_knowledge_graph(current_sub_indices, dominant_pollutant, reasoning_text, predicted_aqi_24h):
    """
    Creates a fresh graph, injects expert knowledge, and logs the 24-hour forecast.
    This replaces the old broken subprocess logic.
    """
    print("\n--- Building 24-Hour RDF Knowledge Graph ---")
    
    # 1. Initialize empty graph and add static knowledge 
    # (This ensures the cause/effect data never disappears)
    g = init_graph()
    add_expert_knowledge(g)
    
    # 2. Define the Prediction Node for this 24-hour window
    pred_id = datetime.now().strftime('%Y%m%d_%H%M')
    pred_uri = URIRef(f"{AQ}Forecast_{pred_id}")
    
    g.add((pred_uri, RDF.type, AQ.Prediction))
    
    # Record max predicted AQI over the next 24h to set the "worst-case" expectation
    max_predicted_aqi = max(predicted_aqi_24h) if predicted_aqi_24h else 0
    g.add((pred_uri, AQ.hasPredictedValue, Literal(float(max_predicted_aqi), datatype=XSD.float)))
    
    # Add the AI reasoning from priority_and_dominance.py
    if reasoning_text:
        g.add((pred_uri, AQ.hasContextualInsight, Literal(reasoning_text)))
    
    # 3. Link the current features to the prediction
    for feature_name, value in current_sub_indices.items():
        feat_uri = URIRef(f"{AQ}Feature_{feature_name}")
        g.add((pred_uri, AQ.hasFeatureInput, feat_uri))
        g.add((feat_uri, AQ.hasObservedValue, Literal(float(value), datatype=XSD.float)))
        
        # Add dynamic warning for the dominant pollutant
        if feature_name == dominant_pollutant:
            warning_msg = f"CRITICAL: {feature_name} is the Dominant Pollutant driving the AQI over the next 24 hours."
            g.add((pred_uri, AQ.hasDynamicWarning, Literal(warning_msg)))

    # 4. Save the Graph
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    g.serialize(destination=OUTPUT_FILE, format="turtle")
    print(f"✅ Successfully saved 24-hour forecast to '{OUTPUT_FILE}'\n")


# MANUAL FALLBACK
# =========================================================
if __name__ == "__main__":
    print("--- Initializing RDF Knowledge Engine (Manual Mode) ---")
    g = init_graph()
    add_expert_knowledge(g)
    g.serialize(destination=OUTPUT_FILE, format="turtle")
    print(f"SUCCESS: Baseline Knowledge Graph saved to '{OUTPUT_FILE}'")

#graph extraction

def extract_graph_for_dashboard():

    if not os.path.exists(OUTPUT_FILE):
        return {"nodes": [], "edges": []}

    g = Graph()
    g.parse(OUTPUT_FILE, format="turtle")

    nodes = {}
    edges = []

    for s, p, o in g:

        s_label = str(s).split("#")[-1]
        p_label = str(p).split("#")[-1]

        if s_label not in nodes:
            nodes[s_label] = {"id": s_label, "label": s_label}

        if isinstance(o, Literal):

            literal_id = f"{p_label}_{hash(str(o))}"

            nodes[literal_id] = {
                "id": literal_id,
                "label": str(o)
            }

            edges.append({
                "source": s_label,
                "target": literal_id,
                "label": p_label
            })

        else:

            o_label = str(o).split("#")[-1]

            if o_label not in nodes:
                nodes[o_label] = {"id": o_label, "label": o_label}

            edges.append({
                "source": s_label,
                "target": o_label,
                "label": p_label
            })

    return {
        "nodes": list(nodes.values()),
        "edges": edges
    }