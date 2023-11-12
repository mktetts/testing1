from python_backend.utils.success import Success
from python_backend.utils.error import Error
import python_backend.contract.blockchain
from python_backend.contract.admin import doctorDetailsInstance, prescriptionDetailsInstance
from python_backend.contract.deploy import deploy_contract as deploy
from web3.middleware import geth_poa_middleware
from flask import jsonify, request
import sys

sys.dont_write_bytecode = True

all_prescriptions = []

def add_prescription():
    status = python_backend.contract.blockchain.connected 
    success = False
    
    try:
        print(request.json)
        keys = list(request.json.keys())
        values = list(request.json.values())

        w3 = python_backend.contract.blockchain.w3
        w3.middleware_onion.inject(geth_poa_middleware, layer=0)
        accounts = w3.eth.accounts

        account = accounts[0]

        prescriptionDetailContract = prescriptionDetailsInstance(w3)
        userPassword = values[2] + "@" + "12345"
        transaction_hash = prescriptionDetailContract.functions.addPrescription({
            "name": values[0],
            "email": values[1],
            "disease" : values[2],
            "prescription" : values[3],
            "comments" : values[4]
            

        }
        ).transact({
            "from": account
        })
        
        success = True

       
    except Exception as e:
        print(e)
        return Error("Failed", str(e), 200)
    if success:
        return Success("Success", "Prescription Added Successfully", 200)
    return Error("Failed", "Prescription not Added", 200)

def get_all_prescription():
    all_prescriptions = []
    w3 = python_backend.contract.blockchain.w3
    # w3.middleware_onion.inject(geth_poa_middleware, layer=0)

    prescriptionDetailContract = prescriptionDetailsInstance(w3)

    doctorCount = prescriptionDetailContract.functions.prescriptionCount().call()
    for i in range(doctorCount):     
        all_prescriptions.append(prescriptionDetailContract.functions.prescription(i).call())

    return Success("Success", all_prescriptions, 200)


def get_prescription_details():
    values = list(request.json.values())
    w3 = python_backend.contract.blockchain.w3
    prescriptionDetailContract = prescriptionDetailsInstance(w3)

    doctorCount = prescriptionDetailContract.functions.prescriptionCount().call()
    for i in range(doctorCount):     
        all_prescriptions.append(prescriptionDetailContract.functions.prescription(i).call())
    for i in range(len(all_prescriptions)):
        if(values[0] == all_prescriptions[i][3]):
            result = {
                "PrescribedBy" : all_prescriptions[i][0],
                "DoctorsEmail" : all_prescriptions[i][1],
                "Disease" : all_prescriptions[i][2],
                "Prescription" : all_prescriptions[i][3],
                "Comments" : all_prescriptions[i][4], 
            }
            return Success("Success",result, 200)
    return Error("Failure", "Prescription Not found", 200)