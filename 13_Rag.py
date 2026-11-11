from sentence_transformers import SentenceTransformer
import numpy as np

# 1. Define documents properly as dictionaries with "doc_id" and "text"
documents = [
    {
        "doc_id": "IT-POL-001",
        "text": """Title: Employee Wi-Fi & Network Access Policy
Category: IT Security
All corporate laptops and mobile devices must connect exclusively to the secure 'Corp-Secure-5G' network using WPA3 Enterprise authentication. Personal devices belonging to contractors or employees are restricted to the "Corp-Guest" Wi-Fi network, which requires web-portal login every 24 hours. Connecting unauthorized hardware, such as personal Wi-Fi routers or Ethernet switches, to company wall ports is strictly prohibited and results in immediate revocation of network privileges."""
    },
    {
        "doc_id": "IT-SUP-014",
        "text": """Title: Troubleshooting GlobalProtect VPN Connections
Category: IT Support

If GlobalProtect fails to establish a tunnel (Error 423), follow these resolution steps:
1. Open Task Manager and terminate the process named "PanGPS.exe".
2. Open Windows Services (`services.msc`) and restart the "Palo Alto Networks GPS" service.
3. Verify that your system time is synchronized with time.windows.com. Discrepancies larger than 120 seconds will cause SSL handshake failures.
If the issue persists, submit a P3 ticket via Jira Service Desk under the "Network Access" component."""
    },
    {
        "doc_id": "IT-INF-089",
        "text": """Title: Infrastructure Backup Schedule and Retention
Category: Cloud Engineering
Production PostgreSQL databases are backed up continuously using Write-Ahead Logging (WAL) via pgBackRest, with full snapshots taken daily at 02:00 UTC. Full database snapshots are retained in AWS S3 Standard storage for 30 days before being transitioned to S3 Glacier Flexible Retrieval for 7 years. Disaster recovery (DR) failover drills are conducted quarterly. The Recovery Point Objective (RPO) is strictly set to 5 minutes, and the Recovery Time Objective (RTO) is capped at 4 hours."""
    },
    {
        "doc_id": "IT-SEC-005",
        "text": """Title: Credential Security and SSO Guidelines
Category: Cybersecurity
All employee accounts must be secured using Single Sign-On (SSO) via Okta with FIDO2/WebAuthn hardware security keys (e.g., YubiKey) as the mandatory secondary factor. Passwords for non-SSO local accounts must be at least 16 characters in length, containing uppercase letters, lowercase letters, numbers, and symbols. Passwords must not contain personal identifiers or company brand names. Local credentials automatically expire after 90 days and cannot be reused for 12 consecutive rotations."""
    },
    {
        "doc_id": "HR-PTO-101",
        "text": """Title: Paid Time Off (PTO) and Vacation Accrual
Category: HR Policies

Full-time salaried employees receive 20 days of Paid Time Off (PTO) per calendar year, accrued at a rate of 1.66 days per completed month of service. A maximum of 5 unused PTO days may be carried over into the following calendar year, but these rolled-over days must be used by March 31st or they will be forfeited. PTO requests exceeding 3 consecutive working days must be submitted in BambooHR at least 14 days in advance for manager approval."""
    },
    {
        "doc_id": "HR-BEN-204",
        "text": """Title: Paid Parental Leave Policy
Category: Benefits

The company provides up to 16 weeks of 100% paid Parental Leave to eligible birth parents, non-birth parents, adoptive parents, and foster parents following the birth or placement of a child. Employees become eligible for this benefit after 180 continuous days of full-time employment. Parental leave can be taken continuously or divided into increments of no less than two weeks, provided all leave is completed within 12 months of the qualifying event."""
    },
    {
        "doc_id": "HR-FIN-302",
        "text": """Title: Remote Work Equipment and Home Office Allowance
Category: Employee Benefits

Full-time remote workers are eligible for a one-time setup reimbursement of up to $1,000 for home office furniture, ergonomics, and peripherals (monitors, keyboards, standing desks). Additionally, remote employees receive a monthly stipend of $75 to offset home high-speed internet and utility expenses. Expense receipts must be uploaded to Concur within 45 days of purchase for processing."""
    },
    {
        "doc_id": "HR-ETH-001",
        "text": """Title: Workplace Conduct and Harassment Prevention
Category: Ethics & Compliance

The company maintains a strict zero-tolerance policy against unlawful discrimination, bullying, and sexual harassment. Any employee who experiences or witnesses misconduct is encouraged to report it immediately to their HR Business Partner or via the anonymous Ethics Hotline at ethics.company.com. All reports are investigated within 72 business hours. Retaliation against any individual reporting a violation in good faith is strictly prohibited and results in immediate termination of employment."""
    },
    {
        "doc_id": "FIN-EXP-401",
        "text": """Title: Business Travel and Expenses Guidelines
Category: Corporate Finance

Economy class airfare must be booked for all domestic and international flights lasting less than 8 continuous hours; business class booking requires prior approval from a Vice President. The daily per diem for meals during domestic travel is capped at $75 ($15 breakfast, $20 lunch, $40 dinner). Itemized receipts are required for any single expense item over $25. Expense reports submitted more than 60 days after the expense date will be rejected automatically."""
    },
    {
        "doc_id": "FIN-PROC-012",
        "text": """Title: Purchasing Thresholds and Vendor Contracts
Category: Procurement

Purchases under $2,500 can be executed directly using a corporate credit card without prior PO approval. Purchases between $2,500 and $25,000 require a formal Purchase Order (PO) signed by the Department Manager. Any software or service contract exceeding $25,000 annually must undergo competitive bidding with at least 3 vendor quotes, a security review by Infosec, and final authorization from the Chief Financial Officer."""
    },
    {
        "doc_id": "LEG-AGR-009",
        "text": """Title: Standard Non-Disclosure Agreement (NDA) Scope
Category: Legal

Confidential Information includes all proprietary technical data, source code, trade secrets, customer lists, financial projections, and product roadmaps disclosed by the Disclosing Party. The Receiving Party agrees to maintain confidentiality using the same degree of care used for its own confidential assets, but not less than reasonable care. Confidentiality obligations survive for a period of five (5) years from the date of disclosure, except for trade secrets, which remain protected indefinitely."""
    },
    {
        "doc_id": "LEG-PRIV-102",
        "text": """Title: GDPR & CCPA Consumer Data Requests
Category: Data Compliance

Under the GDPR and CCPA regulations, data subjects have the right to request access to, correction of, or deletion of their Personally Identifiable Information (PII). Data Subject Access Requests (DSARs) must be fulfilled by the Privacy Office within 30 calendar days of receipt. Verification of identity via multi-factor authentication is mandatory before disclosing any user records. All PII deletion operations must cascade to third-party sub-processors within 14 days."""
    },
    {
        "doc_id": "MED-TRI-055",
        "text": """Title: Emergency Room Patient Triage Protocol
Category: Clinical Operations

Patients arriving at emergency intake must be triaged within 10 minutes of arrival using the Emergency Severity Index (ESI) scale from Level 1 (Resuscitation) to Level 5 (Non-urgent). ESI Level 1 indicates immediate life-threatening distress (e.g., cardiac arrest, severe respiratory failure) requiring immediate medical intervention. ESI Level 2 represents high-risk situations, confusion, severe pain, or vital sign instability requiring rapid evaluation within 15 minutes."""
    },
    {
        "doc_id": "MED-SEC-110",
        "text": """Title: Electronic Protected Health Information (ePHI) Security
Category: Healthcare Compliance

All systems storing, processing, or transmitting electronic Protected Health Information (ePHI) must employ AES-256 encryption at rest and TLS 1.3 encryption in transit. Clinical staff must log out of terminal workstations immediately when leaving patient care areas. Sharing clinical accounts or sending patient identifiers via unencrypted text message or personal email is a direct violation of HIPAA regulations and subject to administrative suspension and civil penalties."""
    },
    {
        "doc_id": "SAF-LAB-003",
        "text": """Title: Hazardous Chemical Storage and Handling
Category: Workplace Safety

All flammables must be stored in designated yellow, fire-rated safety cabinets compliant with OSHA standard 1910.106. Acids and bases must be stored in separate dedicated polyethylene containment trays to prevent violent neutralization reactions. Personal Protective Equipment (PPE)—including nitrile gloves, splash goggles, and flame-resistant lab coats—must be worn at all times when handling volatile organic solvents. Chemical spills over 500 mL require immediate evacuation and notification of the Hazmat Response Team."""
    }
]

# 2. Chunking Logic
chunk_size = 100
overlap = 20
step = chunk_size - overlap
all_chunks = []

for doc in documents:
    doc_id = doc["doc_id"]
    text = doc["text"]
    words = text.split() 
    
    start = 0
    while start < len(words):
        chunk_words = words[start : start + chunk_size]
        chunk_text = " ".join(chunk_words)
        
        all_chunks.append({
            "doc_id": doc_id,
            "chunk_text": chunk_text,
            "start_idx": start,
            "end_idx": start + len(chunk_words)
        })
        
        start += step

# Extract chunk texts for embedding
chunk_texts = [chunk["chunk_text"] for chunk in all_chunks]
    
# 3. Encoder & Embedding Setup
encoder = SentenceTransformer("all-MiniLM-L6-v2")
chunks_embeddings = encoder.encode(
    chunk_texts,
    normalize_embeddings=True
)

# 4. Query & Retrieval
query = input("Write query: ")
query_embedding = encoder.encode(
    query,
    normalize_embeddings=True
)
sim = query_embedding @ chunks_embeddings.T

sorted_ar = np.argsort(sim)[::-1]
sort = sorted_ar[:3]

print("\n--- Top Matching Chunks ---")
for index in sort:
    print(f"\nDoc ID: {all_chunks[index]['doc_id']}")
    print(all_chunks[index]["chunk_text"])
    
    
context_string = ""
for index in sort:
    chunk = all_chunks[index]
    # Appending both doc_id and chunk text so the LLM knows the source reference
    context_string += f"\n[Doc ID: {chunk['doc_id']}]\n{chunk['chunk_text']}\n"

# 6. Create the final prompt using an f-string
final_prompt = f"""You are an assistant answering questions using the supplied context.

Context:
{context_string}

Question: {query}

Instructions:
- Answer using the context.
- If the context does not contain the answer, say so.
- Do not invent facts.
- Mention which source document supports the answer."""

print("\n--- Final Generated Prompt for LLM ---")
print(final_prompt)