import pandas as pd

print(pd.__version__)


def replaceParts(parts, target, replacement):

    for i in range(len(parts)):
        if parts[i] == target:
            parts[i] = replacement

df = pd.read_excel("ngrams_frequencies_withNames.xlsx", sheet_name=None)

bigram_freqs_counter = {}


word = "hi_how_are_you"

parts = word.split("_")

#print(parts)



#Lamed_Waw


#print(df["new_list"])


for key, row in df["new_list"].iterrows():


    parts = row[1].split("_")

    replaceParts(parts, "Tsadi", "Tsadi-medial")


    for i in range(len(parts) - 1):

        pair = parts[i] + "_" + parts[i + 1]
        currentN = len(parts)
        currentCount = row[2]

        if pair not in bigram_freqs_counter:
            bigram_freqs_counter[pair] = (currentCount, currentN)
        else:
            f = bigram_freqs_counter[pair]
            previousCount = f[0]
            previousN =  f[1]

            #if n < len(parts):
            if currentN < previousN:
                bigram_freqs_counter[pair] = (previousCount + currentCount, currentN)
                #print("problem: for an n and m > n,  m freq  > n freq")
                #print(pair)


        #print(pair)

    #print("--- next --- \n")







created_hebrew_bigram_freqs = {key : bigram_freqs_counter[key][0] for key in bigram_freqs_counter}

#print(created_hebrew_bigram_freqs)



#line 8: Lamed_Waw_Alef 361
#line 24: Lamed_Waw 185

#line 135: Alef_Lamed_Waw_He_Yod_Mem 41

#line 2550: Alef_Lamed_Waw 3