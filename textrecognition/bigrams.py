

def bigram_probs_from_bigram_frequencies(bifreqs, letters):


    freqs = {c: 0 for c in letters}

    for key in bifreqs:

        freqs[key.split("_")[0]] += bifreqs[key]
        if bifreqs[key] == 0:
            bifreqs[key] = 1

    bigrams = {key: bifreqs[key] / freqs[key.split("_")[0]] for key in bifreqs}

    return freqs, bigrams





