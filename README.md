# README for Clinical Data Standardization and FHIR Conversion

This repository contains a Python script designed to extract, standardize, and convert clinical data into FHIR (Fast Healthcare Interoperability Resources) format. The script processes doctor's notes, lab reports, and genomic data, and then structures this information into a standardized JSON format. Finally, it converts this structured data into FHIR-compliant resources.

## Features

1. **Clinical Entity Extraction**:
   - Extracts clinical entities (e.g., diagnoses, medications, procedures) from doctor's notes using a pre-trained biomedical Named Entity Recognition (NER) model.
   - Groups extracted entities by type (e.g., "Disease", "Drug", "Procedure").

2. **Lab Report Extraction**:
   - Parses lab reports to extract test names, values, units, and status (e.g., "Normal", "High").
   - Handles common lab report formats and extracts relevant information.

3. **Demographic Data Extraction**:
   - Extracts demographic information such as age, sex, location, and ethnicity from text using a combination of NER and regular expressions.

4. **Genomic Data Extraction**:
   - Processes genomic data from VCF, BAM, or FASTQ files.
   - Extracts genomic variants, aligned reads, or sequence reads depending on the file type.

5. **Data Combination**:
   - Combines extracted clinical entities, lab results, demographic data, and genomic variants into a single structured JSON object.

6. **FHIR Conversion**:
   - Converts the structured JSON data into FHIR-compliant resources (e.g., Patient, Condition, DiagnosticReport, MolecularSequence).
   - Generates a FHIR Bundle containing all the resources.

## Requirements

- Python 3.x
- Libraries:
  - `spacy`
  - `transformers`
  - `fhir.resources`
  - `cyvcf2`
  - `pysam`
  - `biopython`

## Installation

1. **Install Python Libraries**:
   ```bash
   pip install spacy transformers fhir.resources cyvcf2 pysam biopython
   ```

2. **Download SpaCy Model**:
   ```bash
   python -m spacy download en_core_web_md
   ```

## Usage

1. **Prepare Input Files**:
   - Ensure you have a text file containing doctor's notes (`patient_file_path`).
   - Optionally, provide a genomic data file (`genomic_file_path`) in VCF, BAM, or FASTQ format.

2. **Run the Script**:
   - Modify the `patient_file_path`, `genomic_file_path`, and `genomic_file_type` variables in the script to point to your input files.
   - Run the script:
     ```bash
     python standardization.py
     ```

3. **Output**:
   - The script will output two JSON files:
     - `intermediate_json`: Structured JSON containing extracted clinical, lab, demographic, and genomic data.
     - `fhir_json`: FHIR-compliant JSON bundle containing the standardized data.

## Example

```python
# Example usage
patient_file_path = "path/to/patient_notes.txt"
genomic_file_path = "path/to/genomic_data.vcf"
genomic_file_type = "vcf"

patient_data = read_text_file(patient_file_path)
intermediate_json = combine_all(patient_data, genomic_file_path, genomic_file_type)
fhir_json = convert_to_fhir(intermediate_json)

print(intermediate_json)
print(fhir_json)
```

## Code Structure

- **`extract_clinical_entity(doctor_notes)`**:
  - Extracts clinical entities from doctor's notes using a biomedical NER model.
  - Returns a list of entities and a dictionary grouped by entity type.

- **`extract_lab_reports(lab_records)`**:
  - Parses lab reports and extracts test results.
  - Returns a dictionary of lab results.

- **`extract_demographic_data(text)`**:
  - Extracts demographic information (age, sex, location, ethnicity) from text.
  - Returns a dictionary of demographic data.

- **`extract_genomic_data(file_path, file_type)`**:
  - Processes genomic data from VCF, BAM, or FASTQ files.
  - Returns a list of genomic variants or sequence reads.

- **`combine_all(para, genomic_file=None, genomic_type=None)`**:
  - Combines all extracted data into a single structured JSON object.
  - Returns the structured JSON.

- **`convert_to_fhir(intermediate_json)`**:
  - Converts the structured JSON data into FHIR-compliant resources.
  - Returns a FHIR Bundle in JSON format.

## Notes

- Ensure that the input files (doctor's notes, lab reports, genomic data) are properly formatted.
- The script uses a pre-trained biomedical NER model (`d4data/biomedical-ner-all`) for entity extraction. You can replace it with other models if needed.
- The FHIR conversion is based on the FHIR R4 specification. Ensure that your FHIR server or application supports this version.
.

## Acknowledgments

- The script uses the `d4data/biomedical-ner-all` model from Hugging Face for clinical entity extraction.
- FHIR resources are generated using the `fhir.resources` library.

---

For any questions or issues, please open an issue in the repository or contact the maintainers.
