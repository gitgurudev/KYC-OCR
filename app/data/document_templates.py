"""
Document schema knowledge base.
Each entry describes one ID document type — stored in ChromaDB for RAG retrieval.
"""

DOCUMENT_TEMPLATES = [
    {
        "type": "aadhaar",
        "display_name": "Aadhaar Card",
        "category": "government_id",
        "description": (
            "Aadhaar Card is a 12-digit unique identification number issued by UIDAI (Unique "
            "Identification Authority of India). It contains the holder's name, date of birth, "
            "gender, address, and a 12-digit Aadhaar number printed in xxxx xxxx xxxx format. "
            "The card shows UIDAI logo and may have QR code. Text includes आधार in Hindi. "
            "Address includes city, state, and 6-digit PIN code."
        ),
        "visual_keywords": ["UIDAI", "आधार", "Aadhaar", "12-digit number", "QR code", "Unique Identification"],
        "fields": {
            "full_name":      {"label": "Full Name",       "type": "text"},
            "dob":            {"label": "Date of Birth",   "type": "text", "hint": "DD/MM/YYYY"},
            "gender":         {"label": "Gender",          "type": "text", "hint": "Male / Female / Transgender"},
            "aadhaar_number": {"label": "Aadhaar Number",  "type": "text", "hint": "XXXX XXXX XXXX"},
            "address":        {"label": "Address",         "type": "textarea"},
            "pincode":        {"label": "PIN Code",        "type": "text"},
        },
        "extraction_prompt": (
            "Extract these fields from the Aadhaar Card: "
            "full_name (as printed), dob (DD/MM/YYYY), gender, "
            "aadhaar_number (12-digit in format XXXX XXXX XXXX), "
            "address (complete address), pincode (6-digit). "
            "The Aadhaar number may be partially masked — capture whatever is visible."
        ),
    },
    {
        "type": "pan",
        "display_name": "PAN Card",
        "category": "government_id",
        "description": (
            "PAN (Permanent Account Number) Card is issued by the Income Tax Department of India. "
            "It contains a 10-character alphanumeric identifier in the format AAAAA9999A (5 letters, "
            "4 digits, 1 letter). The card shows the holder's name in CAPITALS, father's name, "
            "and date of birth. It has the Income Tax Department logo and Government of India emblem. "
            "Text includes INCOME TAX DEPARTMENT and GOVT. OF INDIA."
        ),
        "visual_keywords": ["INCOME TAX DEPARTMENT", "PAN", "Permanent Account Number", "GOVT. OF INDIA"],
        "fields": {
            "full_name":     {"label": "Full Name",       "type": "text"},
            "fathers_name":  {"label": "Father's Name",   "type": "text"},
            "dob":           {"label": "Date of Birth",   "type": "text", "hint": "DD/MM/YYYY"},
            "pan_number":    {"label": "PAN Number",      "type": "text", "hint": "AAAAA9999A"},
        },
        "extraction_prompt": (
            "Extract from the PAN Card: "
            "full_name (name of the cardholder, usually in CAPITALS), "
            "fathers_name (father's name printed below the holder's name), "
            "dob (date of birth in DD/MM/YYYY format), "
            "pan_number (10-character code: 5 letters, 4 digits, 1 letter)."
        ),
    },
    {
        "type": "passport",
        "display_name": "Passport",
        "category": "travel_document",
        "description": (
            "Indian Passport is a travel document issued by the Ministry of External Affairs. "
            "It has a machine-readable zone (MRZ) at the bottom with two lines of text. "
            "Contains: passport number (1 letter + 7 digits), surname, given names, nationality (IND), "
            "date of birth, sex, place of birth, issue date, expiry date, and file number. "
            "The cover says Republic of India / भारत गणराज्य."
        ),
        "visual_keywords": ["REPUBLIC OF INDIA", "PASSPORT", "MRZ", "IND", "Ministry of External Affairs"],
        "fields": {
            "surname":          {"label": "Surname",          "type": "text"},
            "given_names":      {"label": "Given Names",      "type": "text"},
            "passport_number":  {"label": "Passport Number",  "type": "text", "hint": "A9999999"},
            "nationality":      {"label": "Nationality",      "type": "text"},
            "dob":              {"label": "Date of Birth",    "type": "text", "hint": "DD/MM/YYYY"},
            "sex":              {"label": "Sex",              "type": "text"},
            "place_of_birth":   {"label": "Place of Birth",  "type": "text"},
            "issue_date":       {"label": "Date of Issue",   "type": "text"},
            "expiry_date":      {"label": "Date of Expiry",  "type": "text"},
            "file_number":      {"label": "File Number",     "type": "text"},
        },
        "extraction_prompt": (
            "Extract from the Passport: "
            "surname, given_names, passport_number (format: 1 letter + 7 digits), "
            "nationality, dob (DD/MM/YYYY), sex (M/F), place_of_birth, "
            "issue_date, expiry_date, file_number. "
            "If MRZ is visible, use it to verify the passport number and dates."
        ),
    },
    {
        "type": "voter_id",
        "display_name": "Voter ID (EPIC Card)",
        "category": "government_id",
        "description": (
            "Voter ID card (EPIC - Electoral Photo Identity Card) is issued by the Election Commission "
            "of India. Contains: voter's name, father's or husband's name, date of birth, "
            "EPIC number (3 letters + 7 digits), address, assembly constituency, and part number. "
            "Has ELECTION COMMISSION OF INDIA text and the ECI logo."
        ),
        "visual_keywords": ["ELECTION COMMISSION OF INDIA", "EPIC", "Electoral Photo Identity Card", "Voter"],
        "fields": {
            "full_name":       {"label": "Full Name",             "type": "text"},
            "relation_name":   {"label": "Father's / Husband's Name", "type": "text"},
            "relation_type":   {"label": "Relation",              "type": "text", "hint": "Father / Husband"},
            "dob":             {"label": "Date of Birth",         "type": "text"},
            "epic_number":     {"label": "EPIC Number",           "type": "text", "hint": "XXX9999999"},
            "address":         {"label": "Address",               "type": "textarea"},
            "constituency":    {"label": "Assembly Constituency", "type": "text"},
        },
        "extraction_prompt": (
            "Extract from the Voter ID (EPIC Card): "
            "full_name (voter's name), relation_name (father's or husband's name), "
            "relation_type (Father or Husband), dob (date of birth), "
            "epic_number (3 letters + 7 digits), address, constituency."
        ),
    },
    {
        "type": "driving_license",
        "display_name": "Driving License",
        "category": "government_id",
        "description": (
            "Indian Driving License is issued by the Regional Transport Office (RTO). "
            "Contains: license number (state code + RTO code + year + serial), "
            "holder's name, address, date of birth, issue date, expiry date, "
            "and vehicle classes (LMV, MCWG, etc.). Has RTO/transport department logo."
        ),
        "visual_keywords": ["DRIVING LICENCE", "RTO", "Transport", "LMV", "MCWG", "Motor Vehicle"],
        "fields": {
            "full_name":       {"label": "Full Name",       "type": "text"},
            "license_number":  {"label": "License Number",  "type": "text"},
            "dob":             {"label": "Date of Birth",   "type": "text"},
            "issue_date":      {"label": "Issue Date",      "type": "text"},
            "expiry_date":     {"label": "Expiry Date",     "type": "text"},
            "address":         {"label": "Address",         "type": "textarea"},
            "vehicle_class":   {"label": "Vehicle Class(es)", "type": "text", "hint": "e.g. LMV, MCWG"},
            "blood_group":     {"label": "Blood Group",    "type": "text"},
        },
        "extraction_prompt": (
            "Extract from the Driving License: "
            "full_name, license_number (state-RTO-year-serial format), "
            "dob (date of birth), issue_date, expiry_date, address, "
            "vehicle_class (all vehicle categories listed), blood_group if visible."
        ),
    },
    {
        "type": "ration_card",
        "display_name": "Ration Card",
        "category": "government_document",
        "description": (
            "Ration Card is issued by the state government's Food & Civil Supplies Department. "
            "Contains: ration card number, card type (AAY/BPL/APL/PHH), head of family name, "
            "address, family members list with their names and ages, "
            "and the name of the fair price shop. Has state government logo."
        ),
        "visual_keywords": ["RATION CARD", "Food and Civil Supplies", "BPL", "APL", "AAY", "Fair Price Shop"],
        "fields": {
            "card_number":     {"label": "Ration Card Number", "type": "text"},
            "card_type":       {"label": "Card Type",          "type": "text", "hint": "AAY / BPL / APL / PHH"},
            "head_of_family":  {"label": "Head of Family",     "type": "text"},
            "address":         {"label": "Address",            "type": "textarea"},
            "family_members":  {"label": "Family Members",     "type": "textarea", "hint": "Names and ages"},
            "issue_date":      {"label": "Issue Date",         "type": "text"},
        },
        "extraction_prompt": (
            "Extract from the Ration Card: "
            "card_number, card_type (AAY/BPL/APL/PHH), "
            "head_of_family (name of the primary cardholder), address, "
            "family_members (all names listed, comma separated), issue_date."
        ),
    },
    {
        "type": "utility_bill",
        "display_name": "Utility / Electricity Bill",
        "category": "address_proof",
        "description": (
            "Utility bills (electricity/water/gas/telephone) are issued by service providers. "
            "They contain: consumer name, consumer number/account number, service address, "
            "billing period, bill date, due date, total amount payable, "
            "meter readings, and the service provider's name and contact."
        ),
        "visual_keywords": ["BILL", "Consumer", "Account Number", "Amount Payable", "Due Date", "Electricity", "MSEB", "BESCOM", "BSES"],
        "fields": {
            "consumer_name":    {"label": "Consumer Name",      "type": "text"},
            "consumer_number":  {"label": "Consumer / Account No.", "type": "text"},
            "service_address":  {"label": "Service Address",    "type": "textarea"},
            "bill_date":        {"label": "Bill Date",          "type": "text"},
            "due_date":         {"label": "Due Date",           "type": "text"},
            "billing_period":   {"label": "Billing Period",     "type": "text"},
            "amount_payable":   {"label": "Amount Payable",     "type": "text"},
            "service_provider": {"label": "Service Provider",   "type": "text"},
        },
        "extraction_prompt": (
            "Extract from the Utility Bill: "
            "consumer_name, consumer_number (account/consumer ID), "
            "service_address (address of service connection), "
            "bill_date, due_date, billing_period, amount_payable, service_provider (company name)."
        ),
    },
    {
        "type": "bank_passbook",
        "display_name": "Bank Passbook / Statement",
        "category": "financial_document",
        "description": (
            "Bank passbook or statement issued by a bank. Contains: account holder name, "
            "account number, IFSC code, branch name and address, account type (savings/current), "
            "and transaction history. Has the bank's logo and name."
        ),
        "visual_keywords": ["BANK", "IFSC", "Account Number", "Savings", "Current", "Branch", "Passbook"],
        "fields": {
            "account_holder":  {"label": "Account Holder Name", "type": "text"},
            "account_number":  {"label": "Account Number",      "type": "text"},
            "ifsc_code":       {"label": "IFSC Code",           "type": "text"},
            "bank_name":       {"label": "Bank Name",           "type": "text"},
            "branch_name":     {"label": "Branch",              "type": "text"},
            "branch_address":  {"label": "Branch Address",      "type": "textarea"},
            "account_type":    {"label": "Account Type",        "type": "text", "hint": "Savings / Current"},
        },
        "extraction_prompt": (
            "Extract from the Bank Passbook/Statement: "
            "account_holder (full name), account_number, ifsc_code, "
            "bank_name, branch_name, branch_address, account_type."
        ),
    },
]

# Flat list for embedding generation
def get_all_templates() -> list[dict]:
    return DOCUMENT_TEMPLATES

def get_template_by_type(doc_type: str) -> dict | None:
    for t in DOCUMENT_TEMPLATES:
        if t["type"] == doc_type:
            return t
    return None
