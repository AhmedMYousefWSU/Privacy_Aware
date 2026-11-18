"""VISPR-redactions privacy attributes (24 localizable classes used in the CVPR'18 release, plus extras
present in the official question file). Questions are taken from the official
resources/attr_id_to_question.tsv of github.com/tribhuvanesh/visual_redactions (HTML stripped)."""

ATTRIBUTES = [  # (attr_id, name, yes/no question)
    ('a7_fingerprint', 'Fingerprint', "Is someone's Fingerprint (such as a close-up shot of a finger displaying ridges or finger impressions on paper) visible in the image?"),
    ('a8_signature', 'Signature', "Is someone's Signature visible in the image?"),
    ('a18_ethnic_clothing', 'Traditional clothing', 'Does someone appear in Traditional clothing (such as dirndl, saree, kilts, kimono) in the image?'),
    ('a24_birth_date', 'Date of Birth', "Is someone's Date of Birth (year, month, or date) visible in the image?"),
    ('a26_handwriting', 'Handwriting', 'Is Handwritten text visible in the image?'),
    ('a29_ausweis', 'National Identification', "Is someone's National Identity Card (Green Card, Residence Permit, European ID, etc.) visible in the image?"),
    ('a30_credit_card', 'Credit Card', "Is someone's Credit Card or related details(such as partial Credit Card number, MasterCard/Visa, expiry date, bank name, etc.) visible in the image?"),
    ('a31_passport', 'Passport', "Is someone's Passport (either front cover or any page of a passport) visible in the image?"),
    ('a32_drivers_license', 'Drivers License', "Is someone's Drivers License or related information visible in the image?"),
    ('a33_student_id', 'Student ID', "Is someone's Student ID card visible in the image?"),
    ('a35_mail', 'Mail', "Is someone's Letter, Mail, Mail content or envelope visible in the image?"),
    ('a37_receipt', 'Receipts', 'Is a Receipt (or a document displaying a financial transaction) visible in the image?'),
    ('a38_ticket', 'Tickets', 'Is a Ticket (like for a concert, sport event, flight, travel, etc) visible in the image?'),
    ('a39_disability_physical', 'Physical disability', 'Is someone Physically disabled in the image?'),
    ('a43_medicine', 'Medical History', "Is someone's Medical History (e.g., pill type, medicine names, prescriptions, etc) visible in the image?"),
    ('a49_phone', 'Phone number', 'Is a Phone number (of any person, business, organization, etc) visible in the image?'),
    ('a70_education_history', 'Education history', "Is someone's Education Information (such as certificates, school/college they graduated from or its logo, grade transcripts, etc) visible in the image?"),
    ('a73_landmark', 'Visited Landmark or Street Sign', 'Is a Landmark Name, Sign or a Landmark visible in the image?'),
    ('a82_date_time', 'Date/Time of Activity', 'Is a Date or Time information(Year, Month, Date or Time) visible in the image?'),
    ('a85_username', 'Username', 'Is a Username (e.g., Twitter/Flickr username) visible in the image?'),
    ('a90_email', 'Email address', 'Is an E-mail address visible in the image?'),
    ('a105_face_all', 'Face', "Is someone's Face visible in the image?"),
    ('a106_address_current_all', 'Current Address', 'Is Location Information (address, street sign, GPS co-ordinates) indicating a location visible in the image?'),
    ('a107_address_home_all', 'Home address', 'Is a Home or Postal Address (either Country, City, State, Street, etc.) visible in the image?'),
    ('a108_license_plate_all', 'License Plate', 'Is a Vehicle License Plate or Vehicle Registration Information visible in the image?'),
    ('a109_person_body', 'Person Body', 'Is a Person (of any age), including photos of them or any part of a person (such as only feet) visible in the image?'),
    ('a110_nudity_all', 'Brief or Full nudity', 'Is someone Fully/Briefly nude (no clothes, shirtless, topless, in undergarments, bikini, etc) visible in the image?'),
    ('a111_name_all', 'Name', "Is a Person's Name visible in the image?"),
]

ATTR_IDS = [a for a, _, _ in ATTRIBUTES]
ATTR_NAME = {a: n for a, n, _ in ATTRIBUTES}
ATTR_QUESTION = {a: q for a, _, q in ATTRIBUTES}
