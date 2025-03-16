# **Secure Health Data Exchange System**
*Federated Learning + Blockchain (Hyperledger)*

note: Please refer to other branches with the various parts of the code. Also note the hyperledge branch uses a starter template (hence the number of commits).

## **📌 Overview**
> Data is the new oil, so there arises a need to store the data securely and train ml models without giving away the data.
> With hyperledger fabric. we can do just that. Just be a part of an organization and store the data in a private data storage.
> With federated learning, train the ml models locally and transfer them to a "super" model to aggregate the weights.
> Transform the data into a standard FHIR format.

## **🚀 Features**
- 🔒 **Secure Health Data Exchange** using **Hyperledger Blockchain**
- 🏥 **Federated Learning** for decentralized AI model training
- 🧠 **AI-driven Medical Insights** (NLP + Structured Analysis)
- 🌐 **FHIR-compliant Interoperability** for medical data standardization

## **🛠️ Tech Stack**
### **Backend**
- FastAPI (Python)
- Hyperledger Fabric (Blockchain)
- Federated Learning (Flower)
- AI Models (Biomedical NER)

### **Frontend**
- React.JS
- Chart.JS
- TailwindCSS

### **Databases & Storage**
- CouchDB (using hyperledger fabric)

## **🔑 Secure Architecture**
- **Blockchain Encryption** (Hyperledger Smart Contracts)
- **Federated Learning Workflow** for AI model training without data sharing
- **Data Standardization** using **FHIR & NLP**

## **⚙️ Setup & Installation**
### **Prerequisites**
- Docker & Docker Compose
- Python 3.10+
- Node.js 18+
- MongoDB & IPFS Node

#To run the hyperledger fabric setup
- clone the repo
- go to ```test-network``` directory
- make ```./restart.sh``` as executable by running ```chmod +x ./restart.sh```
- run ```./restart.sh```
  
