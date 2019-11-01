
from textrecognition.english_ngrams import bigrams, char_range
from textrecognition.viterbi import viterbi

english_start_freq = { 't': .15978, 'a': .11682,'o':.07631 ,'i':.07294 ,'s':.06686 ,'w':.05497 ,'c':.05238 ,'b':.04434 ,'p':.04319 ,'h':.04200 ,'f':.04027 ,'m':.03826 ,'d':.03174 ,'r':.02826 ,'e':.02799 ,'l':.02415 ,'n':.02284 ,'g':.01642 ,'u':.01183 ,'v':.00824 ,'y':.00763 ,'j':.00511 ,'k':.00456 ,'q':.00222 ,'x':.00045 ,'z':.00045}



obs = [
 {letter : {'t' : 0.01, 'z': 0.8}.get(letter, 0) for letter in char_range('a', 'z') }         ,
 {letter : {'i' : 0.5}.get(letter, 0) for letter in char_range('a', 'z') }         ,
 {letter : {'g' : 0.5}.get(letter, 0) for letter in char_range('a', 'z') }         ,
 {letter : {'e' : 0.5}.get(letter, 0) for letter in char_range('a', 'z') }         ,
 {letter : {'r' : 0.5}.get(letter, 0) for letter in char_range('a', 'z') }         ,


]



#obs = [ob1, ob2, ob3]
states = list(char_range('a', 'z'))


#for st in states:
 #   print(st)

start_p = english_start_freq
trans_p = bigrams
emit_p = lambda state, obs : obs[state]



viterbi(obs, states, start_p, trans_p, emit_p)

