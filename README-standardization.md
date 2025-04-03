# Clinical Data Standardization and FHIR Conversion READM

This repository contains a Python script that is designed to extract, normalize, and convert clinical data into FHIR format. The script converts doctor's notes, laboratory results, and genomic data and then structures this data in a normalized JSON format. It then converts this structured data into FHIR-compliant resources.

## Features

1. **Clinical Entity Extraction:**
- Extracts clinical entities (e.g., diagnoses, medications, procedures) from physicians' notes from a pre-trained biomedical Named Entity Recognition (NER) model.
- They were organized in type groups (e.g., "Disease", "Drug", "Procedure").

2. **Lab Report Extraction:**
- Pulls test names, values, units, and status (e.g., "Normal", "High") out of lab reports.
- Translates standard lab report structures into necessary information.

3. **Demographic Data Extraction:**
- Extracts demographic information such as age, sex, place, and ethnicity from text through regular expressions as well as NER.

4. **Genomic Data Extraction:**
- Imports genomic information from VCF, BAM, or FASTQ files.
- Reads genomic variants, aligned reads, or sequence reads depending on the type of file.

5. **Data Combination:**
- Combines clinical objects extracted, lab data, demographic data, and genomic variants into a single JSON object.

6. **FHIR Conversion:**
- Converts the organized JSON data into FHIR-compliant resources (i.e., Patient, Condition, DiagnosticReport, MolecularSequence).
- Returns a FHIR Bundle of all resources.

## Requirements

- Python 3.x
- Libraries
- `spacy`
- `transformers`
- `fhir.resources`
- `cyvcf2`
- `pysam`
- `biopython`

## Instalation

1. **Install Python Libraries:**
```bash
```
pip install spacy transformers fhir.resources cyvcf2 pysam biopython
```

2. **Download SpaCy Model:**
``bash```
python -m spacy download en_core_web_md
```

## Usage

1. **Prepare Input Files:**
- Ensure that you have a text file named doctor's notes (`patient_file_path`).
- Optionally, provide a genomic data file (`genomic_file_path`) in VCF, BAM, or FASTQ format.

2. **Run the Script:**
- Modify the `patient_file_path`, `genomic_file_path`, and `genomic_file_type` variable in the code to point to your input file paths.
- Run the script:
```bash
python standardization.py
```

3. **Output:**
- The script will print two JSON files:
- `intermediate_json`: JSON formatted data containing extracted genomic, demographic, clinical, and lab information.
- `fhir_json`: FHIR JSON bundle of the standardized data.

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

- Extracts clinical concepts from physician notes using a biomedical NER system.

- Returns a list of entities and an entity type dictionary.

- **`extract_lab_reports(lab_records)`:**

- Reads laboratory reports and derives results from testing.

- Returns a dictionary of lab results.

- **`extract_demographic_data(text)`:**

- Extracts demographic information (age, gender, location, ethnicity) from text.

- Returns a dictionary of population data.

- **`extract_genomic_data(file_path, file_type)`**:

- Imports genomic data from VCF, BAM, or FASTQ files.

- Returns a list of genomic variants or sequence reads.

- **`combine_all(para, genomic_file=None, genomic_type=None)`:**

- Aggregates all the extracted data into a single structured JSON object. - Returns the structured JSON. - **`convert_to_fhir(intermediate_json)`:** - Transforms the structured JSON data into FHIR-compliant resources. - It returns a FHIR Bundle in JSON. ## Notes - Verify that the input files (genomic data, lab reports, doctor's notes) are properly formatted. - The script uses a pre-trained biomedical NER model (`d4data/biomedical-ner-all`) for entity extraction. You can replace it with other models if needed. - FHIR conversion is based on the FHIR R4 standard. Ensure your FHIR application or server supports this version. ## License This project is under the MIT License. See the [LICENSE](LICENSE) file for details. ## Acknowledgments - The script utilizes the `d4data/biomedical-ner-all` Hugging Face model for the extraction of clinical entities. - FHIR resources are built using the `fhir.resources` library. --- For any issues or questions, please open a repository issue or contact the maintainers.
