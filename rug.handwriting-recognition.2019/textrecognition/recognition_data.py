from data import getdata
from recognition_conifg import recognition_config

import types


print("getting pretrain data in config")
pretrain_data = getdata(recognition_config["pretrain_dataset"])

print("getting data in config")
data = getdata(recognition_config["dataset"])


recognition_data = {
    "pretrain_data":  pretrain_data,
    "data" : data
}

