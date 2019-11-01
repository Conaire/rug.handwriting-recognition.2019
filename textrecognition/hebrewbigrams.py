import pandas as pd
import xlrd
import pickle
from textrecognition.bigrams import bigram_probs_from_bigram_frequencies
from textrecognition.recognition_conifg import recognition_config
from textrecognition.hebrew_create_bigram_freqs import created_hebrew_bigram_freqs, replaceParts

df = pd.read_excel("ngrams_frequencies_withNames.xlsx", sheet_name=None)


args = recognition_config

# {
#    "dataset": "../monkbrill2",
 #   "plot": "output/simple_nn_plot.png",
  #  "model": "output/simple_nn.model",
   # "label_bin": "output/simple_nn_lb.pickle",
#}

#print(df)

lb = pickle.loads(open(args["label_bin"], "rb").read())


labels = lb.classes_




hebrew_bifreqs = created_hebrew_bigram_freqs

hebreq_start_freqs = {label : 0 for label in labels}





for index, row in df["new_list"].iterrows():

    row

    parts = row[1].split("_")

    replaceParts(parts, "Tsadi", "Tsadi-medial")

    #if len(parts) == 2:
    #    hebrew_bifreqs[parts[0] + "_" + parts[1]] = row[2]

    if len(parts) > 2:
        hebreq_start_freqs[parts[0]] += row[2]


hebrew_start_freqs_sum = sum(hebreq_start_freqs.values())
#print("all hebrew start freqs " + str(hebrew_start_freqs_sum))

hebrew_start_probs = {label : hebreq_start_freqs[label] / hebrew_start_freqs_sum  for label in labels}


for x in  labels:
    if "final" in x or x == "Mem":
        hebrew_start_probs[x] = 0
    else:
        hebrew_start_probs[x] = 1 / 22


#print("all hebrew start  prob " + str(sum(hebrew_start_probs.values())))



#print(hebrew_bifreqs)
#print(labels)


freqs, hebrew_bigrams = bigram_probs_from_bigram_frequencies(hebrew_bifreqs, labels)


hebrew_states = labels

#print(hebrew_bigrams)



#print(len(hebrew_bigrams))




