import os
import hashids
import hashlib
import binascii
import datetime


"""
Algorithm of hashing patient's id.
1. Split patient's id (maximum of 10 digits) into every 5 digits.
2. Generate hashids for each 5 digits and concatenate them.
3. Add prefix of project identifier.
"""


hashids = hashids.Hashids(salt='secret string for crest project')


def hash_patient_id(patient_id):
    if not patient_id.isdigit():
        print('The patient id is already converted: {}'.format(patient_id))
        return patient_id

    if len(patient_id) > 10:
        raise Exception(
            'The number of digits exceeds 10: {}'.format(patient_id))

    number_padded = patient_id.zfill(10)
    first, second = int(number_padded[:5]), int(number_padded[5:])
    return hashids.encode(first) + '_' + hashids.encode(second)


def anonymize_dicom(dcm):
    today = datetime.date.today()
    if dcm.PatientID.isdigit():
        dcm.PatientID = hash_patient_id(dcm.PatientID)
        # del dcm.PatientName
        dcm.PatientName = 'Hashed at ' + today.strftime('%Y/%m/%d')
    return dcm


def hash_password(passwd):
    salt = os.urandom(16)
    digest = hashlib.pbkdf2_hmac(
        'sha256', str.encode(passwd), salt, 100000
    )
    hpass = salt + digest
    return binascii.hexlify(hpass).decode()


def verify_password(passwd, hpass):
    bin_hpass = binascii.unhexlify(hpass)
    salt = bin_hpass[:16]
    digest = bin_hpass[16:]
    digest_target = hashlib.pbkdf2_hmac(
        'sha256', str.encode(passwd), salt, 100000
    )
    return digest == digest_target
