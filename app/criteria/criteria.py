import re

criteria_dict = {
    "PICO": ["Population", "Intervention", "Comparison", "Outcome"],
    "PICOS": ["Population", "Intervention", "Comparison", "Outcome", "Study Design"],
    "PICOTS": ["Population", "Intervention", "Comparison", "Outcome", "Study Design", "Timeframe"],
    "SPIDER": ["Sample", "Phenomenon of Interest", "Design", "Evaluation", "Research Type"],
    "PEO": ["Population", "Exposure", "Outcome"],
    "SPICE": ["Setting", "Perspective", "Intervention", "Comparison", "Evaluation"],
}


CRITERIA_GUIDANCE = {
    "Population": {
        "query": "Study population characteristics: age, diagnosis, sample size, recruitment, inclusion and exclusion criteria.",
        "instruction": "Summarise the participant population, including age range, diagnosis, recruitment method, and any inclusion or exclusion criteria."
    },
    "Intervention": {
        "query": "Experimental treatment details: name, dosage, frequency, procedure.",
        "instruction": "Describe only the treatment or intervention given to the experimental group. Do not include control or comparator details."
    },
    "Comparison": {
        "query": "Control group or comparator: placebo, standard care, or alternative treatment.",
        "instruction": "Describe only the control or comparator condition used in contrast to the intervention."
    },
    "Outcome": {
        "query": "Measured outcomes: primary, secondary, instruments used, timing.",
        "instruction": "List the study's primary and secondary outcomes, and the methods used to measure them if provided."
    },
    "Study Design": {
        "query": "Study design: RCT, cohort, case-control, cross-sectional, qualitative methods.",
        "instruction": "Identify the study design (e.g., RCT, observational, qualitative), and mention randomization or blinding if described."
    },
    "Timeframe": {
        "query": "Study duration: follow-up period, timing of interventions and outcome assessments.",
        "instruction": "Summarise how long the study lasted, including any follow-up period and timing of data collection."
    },
    "Setting": {
        "query": "Study setting: location, institution type, clinical or educational context.",
        "instruction": "Describe where the study took place, including country, institution type, and social or clinical setting."
    },
    "Evaluation": {
        "query": "Evaluation methods: tools, rating scales, analysis techniques used to assess intervention or outcomes.",
        "instruction": "Describe how the intervention or outcomes were evaluated, including instruments or analysis methods."
    },
    "Results": {
        "query": "Main study findings: statistical outcomes, effect size, significance, qualitative themes.",
        "instruction": "Summarise the main findings of the study, including statistical results or qualitative themes if reported."
    },
    "Exposure": {
        "query": "Exposure of interest: condition, risk factor, or experience relevant to outcome.",
        "instruction": "Describe the exposure, condition, or experience under investigation that relates to the outcome."
    },
    "Phenomenon of Interest": {
        "query": "Phenomenon or experience studied: behaviors, perceptions, conditions, or processes of interest.",
        "instruction": "Summarise the central phenomenon or lived experience being explored or analyzed in the study."
    },
    "Sample": {
        "query": "Sample description: participant group characteristics, demographics, inclusion criteria.",
        "instruction": "Summarise the characteristics of the participant sample, including demographics and selection criteria."
    },
    "Perspective": {
        "query": "Stakeholder or perspective focus: patient, clinician, caregiver, organization viewpoint.",
        "instruction": "Describe whose perspective the study considers, such as patients, providers, or community members."
    },
    "Research Type": {
        "query": "Research type: qualitative, quantitative, or mixed methods approach.",
        "instruction": "Specify the type of research conducted, whether qualitative, quantitative, or mixed methods."
    }
}

def parse_llm_screening_output(raw: str, criteria: list[str]):
    """Parse the raw output from LLM screening into a structured format."""
    
    result = {
        "decision": "Unclear",
        "confidence": 0,
        "rationale": "",
        "criteria_matches": {}
    }
    
    decision_match = re.search(r"Decision:\s*(Include|Exclude|Unclear)", raw, re.IGNORECASE)
    confidence_match = re.search(r"Confidence:\s*(\d)", raw)
    rationale_match = re.search(r"Rationale:\s*(.+)", raw, re.DOTALL)

    if decision_match:
        result["decision"] = decision_match.group(1).capitalize()
    if confidence_match:
        result["confidence"] = int(confidence_match.group(1))
    if rationale_match:
        result["rationale"] = rationale_match.group(1).strip()

    for crit in criteria:
        pattern = rf"{crit}:\s*(.+)"
        match = re.search(pattern, raw, re.IGNORECASE)
        result["criteria_matches"][crit] = match.group(1).strip() if match else "N/A"

    return result