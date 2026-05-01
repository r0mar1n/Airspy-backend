import os
from rdflib import Graph, Namespace

GRAPH_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "final_aqi_knowledge_graph.ttl"
)

AQ = Namespace("http://aqi-prediction.org/ontology#")

def generate_explanation_report():

    g = Graph()

    try:
        g.parse(GRAPH_PATH, format="turtle")
    except FileNotFoundError:
        return "Knowledge graph not found. Run AQI pipeline first."

    query = """
        PREFIX aq: <http://aqi-prediction.org/ontology#>
        
        SELECT ?feature ?value ?risk ?source ?mitigation ?vulnerable ?govSolution ?warning ?insight ?maxAqi
        WHERE {
            ?pred a aq:Prediction .

            OPTIONAL { ?pred aq:hasPredictedValue ?maxAqi . }
            OPTIONAL { ?pred aq:hasContextualInsight ?insight . }
            OPTIONAL { ?pred aq:hasDynamicWarning ?warning . }

            ?pred aq:hasFeatureInput ?featURI .
            ?featURI aq:hasObservedValue ?value ;
                     aq:hasHealthRisk ?risk ;
                     aq:hasLikelySource ?source .

            OPTIONAL { ?featURI aq:mitigationAdvice ?mitigation . }
            OPTIONAL { ?featURI aq:affectedGroup ?vulnerable . }
            OPTIONAL { ?featURI aq:governmentRecommendation ?govSolution . }

            BIND(STRAFTER(STR(?featURI), "Feature_") AS ?feature)
        }
    """

    results = list(g.query(query))

    if not results:
        return "Graph empty or query failed."

    output = []
    output.append("="*70)
    output.append("🌍 AI AIR QUALITY FORECAST & EXPLAINABILITY REPORT (24-HOUR)")
    output.append("="*70)

    first_row = results[0]
    max_aqi = round(float(first_row.maxAqi)) if first_row.maxAqi else "N/A"

    output.append("\n📋 EXECUTIVE SUMMARY:")
    output.append(f"• Worst-Case AQI Expected (Next 24h): {max_aqi}")

    if first_row.insight:
        output.append(f"• AI Atmospheric Reasoning: {first_row.insight}")

    unique_warnings = set([str(row.warning) for row in results if row.warning])
    if unique_warnings:
        output.append("\n🚨 CRITICAL ALERTS:")
        for w in unique_warnings:
            output.append(f"[!] {w}")

    output.append("\n" + "-"*70)
    output.append("📊 DETAILED POLLUTANT BREAKDOWN & PROTOCOLS")
    output.append("-"*70)

    processed_features = set()

    for row in results:
        if row.feature in processed_features:
            continue
        processed_features.add(row.feature)

        val = round(float(row.value), 2)

        output.append(f"\n🔹 {row.feature.upper()} (Current Sub-Index: {val})")
        output.append(f"• Source: {row.source}")
        output.append(f"• Health Risk: {row.risk}")

        if row.vulnerable:
            output.append(f"• Vulnerable: {row.vulnerable}")
        if row.mitigation:
            output.append(f"• Citizen Advice: {row.mitigation}")
        if row.govSolution:
            output.append(f"• GOV PROTOCOL: {row.govSolution}")

    output.append("\n" + "="*70)

    return "\n".join(output)


if __name__ == "__main__":
    print(generate_explanation_report())