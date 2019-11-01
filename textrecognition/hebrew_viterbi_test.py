from statistics import mean

from hebrewbigrams import hebrew_start_probs, hebrew_states, hebrew_bigrams
from viterbi import viterbi
from textrecognition.document_converter import name_to_chr
from document_converter import chr_to_name




#word in document is זכורה


word = "זכורה"

letters = list(word)

letters_name = [chr_to_name[letter] for letter in letters]

#for x in range(len(letters)):
 #   print("is {} ok".format(letters[x]))

print(letters_name)

obs = [
    {letter : {letters_name[0] : 0.3}.get(letter, 0) for letter in hebrew_states }         ,
    {letter : {hebrew_states[9] : 0.4, letters_name[1] : 0.4}.get(letter, 0) for letter in hebrew_states }         ,
    {letter : {letters_name[2] : 0.3}.get(letter, 0) for letter in hebrew_states }         ,
    {letter : {letters_name[3] : 0.3}.get(letter, 0) for letter in hebrew_states }         ,
    {letter : {letters_name[4] : 0.3}.get(letter, 0) for letter in hebrew_states }         ,



]


#print(obs[1])


states = hebrew_states


#for st in states:
 #   print(st)

start_p = hebrew_start_probs
trans_p = hebrew_bigrams
emit_p = lambda state, obs : obs[state]



output = viterbi(obs, states, start_p, trans_p, emit_p)


print(output)

#for i in range(len(hebrew_states)):
 #   print("{} {} vs {}", i )

#print(sorted(name_to_chr.keys()) == sorted(hebrew_states))
#print(trans_p)
#print(mean(trans_p.values())/ 4)
print(hebrew_states[9] )