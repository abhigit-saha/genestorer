python -m spacy download en_core_web_md

import spacy
from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification
from huggingface_hub import login

def extract_clinical_entity(doctor_notes=None):

  #login(token=token)

  # Load SpaCy model
  nlp = spacy.load("en_core_web_md")

  tokenizer = AutoTokenizer.from_pretrained("d4data/biomedical-ner-all")
  model = AutoModelForTokenClassification.from_pretrained("d4data/biomedical-ner-all")

  # Create the NER pipeline
  medical_ner = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple")  # Use device=0 for GPU

  entities = medical_ner(doctor_notes)

  # Group clinical entities by type
  clinical_data = {}
  for entity in entities:
      entity_type = entity['entity_group']
      entity_text = entity['word']
      if entity_type not in clinical_data:
          clinical_data[entity_type] = []
      clinical_data[entity_type].append(entity_text)

  return entities,clinical_data

  # Load SpaCy model
  nlp = spacy.load("en_core_web_md")

  tokenizer = AutoTokenizer.from_pretrained("d4data/biomedical-ner-all")
  model = AutoModelForTokenClassification.from_pretrained("d4data/biomedical-ner-all")

  # Create the NER pipeline
  medical_ner = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple")  # Use device=0 for GPU

  entities = medical_ner(doctor_notes)

  # Group clinical entities by type
  clinical_data = {}
  for entity in entities:
      entity_type = entity['entity_group']
      entity_text = entity['word']
      if entity_type not in clinical_data:
          clinical_data[entity_type] = []
      clinical_data[entity_type].append(entity_text)

  return entities,clinical_data

def extract_lab_reports(lab_records):
  import re

  # Extract lab results
  lab_results = {}
  pattern = r"([\w\s]+):\s*([\d,.]+)\s*([\w/%]+)(?:\s*\((Low|Normal|High|Abnormal|Critical)\))?"

  for line in lab_records.split("\n"):
        matches = re.findall(pattern, line)
        for match in matches:
            test_name, value, unit, status = match
            test_name = test_name.strip()

            test_name = re.sub(r"^(and\s+|Lab results show\s*)", "", test_name, flags=re.IGNORECASE)

            # If status is missing, assign "Unknown"
            if not status:
                status = "Unknown"

            lab_results[test_name] = {
                "value": value.strip(),
                "unit": unit.strip(),
                "status": status.strip()
            }
  return lab_results

def extract_demographic_data(text):
  import re

  demographics = {}
  for entity in extract_clinical_entity(text)[0]:
      if entity["entity_group"] == "Age":
          demographics["age"] = entity["word"]
      elif entity["entity_group"] == "Sex":
          demographics["sex"] = entity["word"]

  # If age or sex is not detected by the NER model, use regex
  if "age" not in demographics:
      age_match = re.search(r"(\d+)\s*(?:yrs?|years?)\s*old", text, re.IGNORECASE)
      demographics["age"] = age_match.group(1) if age_match else None

  if "sex" not in demographics:
      sex_match = re.search(r"\b(male|female|man|woman)\b", text, re.IGNORECASE)
      demographics["sex"] = sex_match.group(1) if sex_match else None


  # Extract additional demographic data (e.g., location)
  location_match = re.search(r"Region:\s*([\w\s]+)", text)
  demographics["location"] = location_match.group(1) if location_match else None

  # Extract ethnicity (example: look for common ethnicity terms)
  ethnicity_match = re.search(r"Ethnicity:\s*([\w\s,]+)", text, re.IGNORECASE)
  if ethnicity_match:
        # Split multiple ethnicities by commas and remove extra spaces
        ethnicities = [eth.strip() for eth in ethnicity_match.group(1).split(",")]
        demographics["ethnicity"] = ethnicities if ethnicities else None

  return demographics

!pip install cyvcf2
!pip install pysam
!pip install biopython # Install biopython
from cyvcf2 import VCF
import pysam
from Bio import SeqIO  # For FASTQ

def extract_genomic_data(file_path, file_type):
    genomic_variants = []

    if file_type.lower() == "vcf":
        # Handle VCF format
        '''try:
            vcf_reader =VCF(file_path)
            print("VCF file parsed successfully!")
        except Exception as e:
            print(f"Error parsing VCF: {e}")
            return genomic_variants'''
        vcf_reader = VCF(file_path)

        for variant in vcf_reader:
            genomic_variants.append({
                "chrom": variant.CHROM,
                "pos": variant.POS,
                "id": variant.ID,
                "ref": variant.REF,
                "alt": variant.ALT[0] if variant.ALT else None,
                "gene": variant.INFO.get("GENE", "Unknown"),
                "mutation": f"{variant.REF}>{variant.ALT[0]}" if variant.ALT else "Unknown"
            })

    elif file_type.lower() == "bam":
        # Handle BAM format (Extract aligned reads info)
        bam_reader = pysam.AlignmentFile(file_path, "rb")
        for read in bam_reader:
            genomic_variants.append({
                "query_name": read.query_name,
                "reference_name": read.reference_name,
                "position": read.reference_start,
                "sequence": read.query_sequence
            })
        bam_reader.close()

    elif file_type.lower() == "fastq":
        # Handle FASTQ format (Extract sequence reads)
        for record in SeqIO.parse(file_path, "fastq"):
            genomic_variants.append({
                "id": record.id,
                "sequence": str(record.seq),
                "quality": record.letter_annotations["phred_quality"]
            })

    else:
        raise ValueError("Unsupported file type. Use 'vcf', 'bam', or 'fastq'.")

    return genomic_variants

def combine_all(para,genomic_file=None,genomic_type=None):
  import json

  entities, clinical_data = extract_clinical_entity( para)

    # Remove "Age" and "Sex" from clinical entities
  filtered_clinical_entities = [
        {"word": e['word'], "entity": e['entity_group']}
        for e in entities if e['entity_group'] not in ["Age", "Sex"]]



    # Combine all data
  structured_data = {
      "demographics": extract_demographic_data(para),
      "clinical_entities": filtered_clinical_entities,
      "lab_results": extract_lab_reports(para),
      "genomic_variants": extract_genomic_data(genomic_file, genomic_type),
      #"miscellaneous": miscellaneous
  }
  '''if genomic_type !=None:  # This line is added
      structured_data["genomic_variants"] = extract_genomic_data(genomic_file, genomic_type)
  else:
      structured_data["genomic_variants"] = []'''

  # Convert to JSON
  structured_json = json.dumps(structured_data, indent=4)

  return structured_json

import json
import uuid
from datetime import datetime

def convert_to_fhir(intermediate_json):
    data = json.loads(intermediate_json)

    # Generate unique IDs
    patient_id = str(uuid.uuid4())

    demographics_data = data.get("demographics") or {}

    age = demographics_data.get("age")
    current_year = datetime.utcnow().year
    birth_year = current_year - age if isinstance(age, int) else 1900

    # Convert Demographics to FHIR Patient Resource
    patient_resource = {
        "resourceType": "Patient",
        "id": patient_id,
        #"name": [{"text": data["demographics"].get("name", "Unknown")}],
        "gender": data.get("demographics", {}).get("sex", "unknown"),
        "birthDate": f"{birth_year}-01-01"
    }

    # Convert Clinical Entities to FHIR Condition Resource
    conditions = []
    for entity_dict in data.get("clinical_entities", []):
        entity = entity_dict.get("entity", "Unknown")
        value = entity_dict.get("word", "Unknown")
        conditions.append({
            "resourceType": "Condition",
            "id": str(uuid.uuid4()),
            "subject": {"reference": f"Patient/{patient_id}"},
            "code": {"text": value},
            "category": [{"text": entity}],
            "recordedDate": datetime.utcnow().isoformat() + "Z"
        })

    observation_id = str(uuid.uuid4())


    # Convert Lab Results to FHIR DiagnosticReport
    diagnostic_report = {
        "resourceType": "DiagnosticReport",
        "id": str(uuid.uuid4()),
        "status": "final",
        "subject": {"reference": f"Patient/{patient_id}"},
        "result": [{"reference": "Observation/{observation_id}"}]
    }

    # Convert Genomic Variants to FHIR MolecularSequence
    molecular_sequences = []
    for variant in data.get("genomic_variants", []):
        molecular_sequences.append({
            "resourceType": "MolecularSequence",
            "id": str(uuid.uuid4()),
            "subject": {"reference": f"Patient/{patient_id}"},
            "coordinateSystem": 1,
            "referenceSeq": {
                "chromosome": {"text": variant.get("chrom", "Unknown")},
                "genomeBuild": "GRCh38",
                "referenceSeqId": {"text": variant.get("gene", "Unknown")}
            },
            "variant": [{
                "start": variant.get("pos", 0),
                "end": variant.get("pos", 0) + len(variant.get("ref", "")) - 1,
                "observedAllele": variant.get("alt", "Unknown"),
                "referenceAllele": variant.get("ref", "Unknown")
            }]
        })
    # Final FHIR Bundle
    fhir_bundle = {
        "resourceType": "Bundle",
        "type": "collection",
        "entry": [{"resource": patient_resource}] +
                 [{"resource": cond} for cond in conditions] +
                 [{"resource": diagnostic_report}] +
                 [{"resource": seq} for seq in molecular_sequences]
    }

    return json.dumps(fhir_bundle, indent=4)

def read_text_file(file_path):
    with open(file_path, 'r') as file:
        return file.read()


patient_data = read_text_file(patient_file_path)

intermediate_json = combine_all(patient_data,genomic_file_path,genomic_file_type)

print(intermediate_json)

fhir_json = convert_to_fhir(intermediate_json)

print(fhir_json)
