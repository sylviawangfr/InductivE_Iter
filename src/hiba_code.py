import random
from transformers import pipeline, BertTokenizer
from sentence_transformers import SentenceTransformer, util
import csv
import numpy as np
import timeit
import sys

#FUNCTIONS
def vowel(ch):
    if(ch == 'a' or ch == 'e' or ch == 'i' or ch == 'o' or ch == 'u' or ch == 'A'
            or ch == 'E' or ch == 'I' or ch == 'O' or ch == 'U'):
        return True
    else:
        return False

def readinput(path):

    entities = []
    with open(str(path), 'r', encoding='UTF8') as fr:
        for line in fr:
            e = line.rstrip()
            if len(e) == 0:
                continue
            entities.append(e)
    return entities

def effify(non_f_str: str):
    return eval(f'f"""{non_f_str}"""')

def sortdic(d):
    d_sorted = sorted(d.items(), key=lambda x: x[1], reverse=True)

    return d_sorted

def getpeerembeddings(path, total):
    PEER_GROUPS = {}
    #economist	physicist,politician,mathematician,economics,engineer,geographer,historian,diplomat,biologist,geologist,biochemist,lawyer,psychologist,agronomist,philosopher,novelist,accountant,inventor,actor,architect,banker,painting,anthropologist,professor,poet,ecologist,illustrator,astronomer,researcher,critic,educator,fiscal policy,composer,soldier
    with open(path, 'r', encoding='UTF8') as fr:
        for line in fr:
            line = line.rstrip()
            line = line.replace("_", " ")
            parts = line.split("\t")
            if len(parts) < 2:
                continue
            main_entity = parts[0]
            peers = parts[1].split(",")
            PEER_GROUPS[main_entity] = peers[:total]
    return PEER_GROUPS

def gethypernyms(path):
    hypers = {}
    #aa	type	0.537823
    with open(path, 'r', encoding='UTF8') as fr:
        for line in fr:
            line = line.rstrip()
            parts = line.split("\t")
            if len(parts) < 3:
                continue
            main_entity = parts[0]
            hypernym = parts[1]
            if hypernym == "thing" or hypernym == "factor" or hypernym == "item":
                continue
            if main_entity not in hypers:
                l = []
                l.append(hypernym)
                hypers[main_entity] = l
            elif len(hypers[main_entity]) < 20:
                l = hypers[main_entity]
                l.append(hypernym)
                hypers[main_entity] = l
    return hypers


def loadoriginalkg(path):

    KG_count = {}
    concepts = {}
    with open(path, 'r', encoding='UTF8') as fr:
        #aa,primary,aa,HasProperty,white,aa,be,white,0.15164147483749477,0.37924222293548027,[]
        reader = csv.reader(fr)
        for parts in reader:

            if len(parts) < 10: #erroneous row
                continue
            concepts[parts[0]]=1
            p = parts[3].rstrip()
            o = parts[4].rstrip()
            if p+o in KG_count:
                KG_count[p+o] = KG_count[p+o] +1
            else:
                KG_count[p+o] = 1
    for key in KG_count:
        KG_count[key] = float(KG_count[key])/float(len(concepts))
    return KG_count

def loadkg(path, th,originalkg):

    KG = {}
    KG_PO = {}
    triple_as_text = {}
    FREQ = {}
    TYPI = {}
    with open(path, 'r', encoding='UTF8') as fr:
        #aa,primary,aa,HasProperty,white,aa,be,white,0.15164147483749477,0.37924222293548027,[]
        reader = csv.reader(fr)
        for parts in reader:

            if len(parts) < 10: #erroneous row
                continue

            s = parts[2].rstrip()
            p = parts[3].rstrip()
            o = parts[4].rstrip()
            if originalkg[p+o] >= th:
                continue
            o = o.replace('\"', "")
            o = o.replace(',', '')
            s_open = parts[5].rstrip()
            p_open = parts[6].rstrip()
            o_open = parts[7].rstrip()
            o_open = o_open.replace('\"', "")
            o_open = o_open.replace(',', '')
            frequency = parts[8].rstrip()
            typicality = parts[9].rstrip()

            #if p == "IsA": #skipping type information (used to collect peers)
            #    continue

            triple = s + ", " + p + ", " + o
            triple_open = s_open + " " + p_open + " " + o_open + ". "
            PO = p + ", " + o
            PO_open = p_open + " " + o_open + ". "

            FREQ[str(s+PO)] = frequency
            TYPI[str(s+PO)] = typicality
            triple_as_text[triple] = triple_open #triple to text
            KG_PO[PO] = PO_open #PO to text

            if s in KG:
                l = KG[s]
                l.append(PO)
                KG[s] = l
            else:
                l = []
                l.append(PO)
                KG[s] = l

    return KG, KG_PO, triple_as_text, FREQ, TYPI

def thresholding_positives(KG,SCORES, k, lamb):
    for s in KG:
        temp = {}
        POs = KG[s]
        for PO in POs:
            sc = SCORES[s+PO]
            try:
                if float(sc) <= lamb:
                    continue
            except Exception:
                continue
            temp[PO]=sc
        temp_list=sortdic(temp)
        triples = []
        for t in temp_list:
            triples.append(t[0])
        if len(triples) < k:
            KG[s] = triples
        else:
            KG[s] = triples[:k]
    return KG

def collectcandidates(e,positive, KG, ordered_members, total):
    #with initial trival validation
    emotional  = ["amazing","great", "good", "wonderful", "bad", "boring", "different", "lovely", "beautiful", "enjoyable", "decent", "less", "more", "most", "fastest", "slowest", "best", "tasteless", "shy"]
    candidate_negations = {}
    validation_details = {}
    details = {}
    unique_details = {}
    hard = {}
    for peer in ordered_members:
        if e == peer:
            continue
        peer_info = []
        if peer in KG:
            peer_info = KG[peer]
        for t in peer_info:
            if t in positive:
                continue
            skip = False
            for em in emotional:
                if em in t:
                    skip = True
                    break
            if skip == True:
                continue
            if t in candidate_negations:
                candidate_negations[t] = candidate_negations[t] + 1
                details[t] = details[t] + ", " + peer
                unique_details[t].append(peer)
            else:
                candidate_negations[t] = 1
                details[t] = peer
                l = []
                l.append(peer)
                unique_details[t] = l
    #normalize by peers
    for candidate in candidate_negations:
        hard[candidate] = candidate_negations[candidate]
        candidate_negations[candidate] = candidate_negations[candidate]/(total)
        validation_details[candidate] = ""
    return candidate_negations, details, hard, validation_details, unique_details

def paraphrase_validation(candidates, KG_PO, positive_info, paraphraser, validation_details,sim_thresh):
    candidate_sentences = []
    sentence_to_triple = {}

    for c in candidates:
        sentence = KG_PO[c].rstrip()
        sentence= sentence.replace('.','')
        sentence= sentence.replace('be ','')
        candidate_sentences.append(sentence)
        sentence_to_triple[sentence] = c

    positives = []
    for pos in positive_info:
        sentence = KG_PO[pos].rstrip()
        sentence = sentence.replace('.','')
        sentence = sentence.replace('be ','')
        positives.append(sentence)
        sentence_to_triple[sentence] = pos

    if len(candidate_sentences) == 0 or len(positives) == 0:
        return candidates, validation_details
    embeddings1 = paraphraser.encode(candidate_sentences, convert_to_tensor=True)
    embeddings2 = paraphraser.encode(positives, convert_to_tensor=True)

    cosine_scores = util.cos_sim(embeddings1, embeddings2)

    scores_ARR = cosine_scores.cpu().numpy()
    candidate_sentences_ARR = np.array(candidate_sentences)
    positives_ARR = np.array(positives)

    for i in range(len(candidate_sentences_ARR)):
        for j in range(len(positives_ARR)):
            sc = scores_ARR[i][j]
            s = str(candidate_sentences_ARR[i])
            p = str(positives_ARR[j])
            key = sentence_to_triple[s]
            key2 = sentence_to_triple[p]
            if sc < sim_thresh or key not in candidates:
                #ALLCHECKS.write(s + "\t" +str(key) + "\t"+ p + "\t"+ str(key2)+ "\t" + str(sc) + "\tPARAPHRASE\n")
                #ALLCHECKS.flush()
                if key in validation_details:
                    validation_details[key] = validation_details[key] + " [paraphrase (" + str(s) + ") (" + str(p) + ")] " + str(sc)
                else:
                    validation_details[key] = " [paraphrase (" + str(s) + ") (" + str(p) + ")] " + str(sc)
                continue
            candidates.pop(key)
            #ALLCHECKS.write(s + "\t" +str(key) + "\t"+ p + "\t"+ str(key2)+ "\t" + str(sc) + "\tPARAPHRASE\n")
            #ALLCHECKS.flush()
    return candidates, validation_details

def lm_subject(subject, candidates, KG_PO, unmasker, paraphraser, stopwords,sim_thresh):
    for c in list(candidates):
        drop_c = False
        token_c = ""
        opentriple = subject + " " + KG_PO[c].rstrip()
        opentriple = opentriple[:-1]
        w = subject.rstrip() #original token
        if len(w.rstrip()) == 0:
            #ALLCHECKS.write("w is empty" + "\t" + "\tLM_SUBJECT\n")
            #ALLCHECKS.flush()
            continue
        if w in stopwords:
            #ALLCHECKS.write("w is stopword" + "\t" + str(w) + "\t" + "\tLM_SUBJECT\n")
            #ALLCHECKS.flush()
            continue
        probe = opentriple
        probe = probe.replace(w, "[MASK]", 1)
        probe = probe.replace(" be ", " is ")
        probe = probe.rstrip() + "."


        predictions = unmasker(effify(probe))

        tokens = []
        original = []
        original.append(w)
        for token_tuple in predictions:
            token = token_tuple['token_str']
            if str(token) in stopwords:
                #ALLCHECKS.write("token is stopword" + "\t" + str(token) + "\tLM_SUBJECT\n")
                #ALLCHECKS.flush()
                continue
            if w == str(token):
                #ALLCHECKS.write("token is exact match" + "\t" + str(token) + "\tLM_SUBJECT\n")
                #ALLCHECKS.flush()
                candidates.pop(c)
                #ALLCHECKS.write(opentriple + "\t" + probe +"\t"+ w + "\t" + token + "\t" + str(1.0) + "\tLM_SUBJECT"+ "\n")
                #ALLCHECKS.flush()
                drop_c = True
                break
            tokens.append(token)

        if drop_c == True:
            continue
        if len(original) == 0 or len(tokens) == 0:
            continue
        embeddings1 = paraphraser.encode(original, convert_to_tensor=True)
        embeddings2 = paraphraser.encode(tokens, convert_to_tensor=True)

        cosine_scores = util.cos_sim(embeddings1, embeddings2)

        scores_ARR = cosine_scores.cpu().numpy()
        original_ARR = np.array(original)
        tokens_ARR = np.array(tokens)

        for i in range(len(original_ARR)):
            max_index = np.argmax(scores_ARR)
            sc = scores_ARR[i][max_index]
            token_c = tokens_ARR[max_index]
            if sc >= sim_thresh:
                #ALLCHECKS.write(opentriple + "\t" + probe +"\t"+ w + "\t" + token_c + "\t" + str(sc) + "\tLM_SUBJECT"+ "\n")
                #ALLCHECKS.flush()
                drop_c = True
                break

        if drop_c == True:
            candidates.pop(c)
            #ALLCHECKS.write(opentriple + "\t" + probe +"\t"+ w + "\t" + token_c + "\t" + str(sc) + "\tLM_SUBJECT"+ "\n")
            #ALLCHECKS.flush()
            continue
    return candidates

def lm_all(subject, candidates, KG_PO, ALLCHECKS, unmasker, paraphraser, stopwords):
    for c in list(candidates):
        successes = 0
        reasons = ""
        opentriple = subject + " " + KG_PO[c].rstrip()
        opentriple = opentriple[:-1]
        words = opentriple.split(' ')
        for i, w in enumerate(words): #w is the original token
            if len(w.rstrip()) == 0:
                #ALLCHECKS.write("w is empty" + "\t" + "\tLM_ALL\n")
                #ALLCHECKS.flush()
                successes = successes + 1
                reasons = reasons + "size zero # "
                continue
            if w in stopwords:
                #ALLCHECKS.write("w is stopword" + "\t" + str(w) + "\t" + "\tLM_ALL\n")
                #ALLCHECKS.flush()
                successes = successes + 1
                reasons = reasons + "stop word # "
                continue
            probe = ""
            index = 0
            isVowel = False
            for x in words:
                if index == i:
                    if vowel(x[0]):
                        isVowel = True
                    probe = probe + " [MASK] "
                else:
                    if x == "be":
                        x = "is"
                    probe = probe + " " + x + " "
                index = index + 1
            probe = probe.rstrip() + "."
            probe = ' '.join(probe.split())
            if "is [MASK]" in probe:
                if isVowel == True:
                    probe = probe.replace('is [MASK]', 'is an [MASK]')
                else:
                    probe = probe.replace('is [MASK]', 'is a [MASK]')

            #start = timeit.default_timer()
            predictions = unmasker(effify(probe))
            #stop = timeit.default_timer()
            #print(str("speed") + "\t" + str(stop - start), flush=True)


            tokens = []
            original = []
            original.append(w)
            sameWord = False
            for token_tuple in predictions:
                token = token_tuple['token_str']
                if str(token) in stopwords:
                    #ALLCHECKS.write("token is stopword" + "\t" + str(token) + "\tLM_ALL\n")
                    #ALLCHECKS.flush()
                    continue
                if w == str(token):
                    #ALLCHECKS.write("token is exact match" + "\t" + str(token) + "\tLM_ALL\n")
                    #ALLCHECKS.flush()
                    successes = successes + 1
                    reasons = reasons + "same token # "
                    sameWord = True
                    break
                tokens.append(token)

            if sameWord == True:
                continue
            if len(tokens) == 0:
                successes = successes + 1
                reasons = reasons + "no eligible tokens # "
                continue
            if len(original) == 0 or len(tokens) ==0:
                #ALLCHECKS.write("original or tokens list empty" + "\tLM_ALL\n")
                #ALLCHECKS.flush()
                continue
            embeddings1 = paraphraser.encode(original, convert_to_tensor=True)
            embeddings2 = paraphraser.encode(tokens, convert_to_tensor=True)

            cosine_scores = util.cos_sim(embeddings1, embeddings2)

            scores_ARR = cosine_scores.cpu().numpy()
            original_ARR = np.array(original)
            tokens_ARR = np.array(tokens)
            similarEnough = False
            for i in range(len(original_ARR)):
                for j in range(len(tokens_ARR)):
                    sc = scores_ARR[i][j]
                    token_c = tokens_ARR[j]
                    if sc >= 0.6:
                        #ALLCHECKS.write(opentriple + "\t" + probe +"\t"+ w + "\t" + token_c + "\t" + str(sc) + "\tLM_ALL"+ "\n")
                        #ALLCHECKS.flush()
                        similarEnough = True
                        successes = successes + 1
                        reasons = reasons + "similar enough " + w + " " +str(tokens_ARR[j]) +" # "
                        break

            if similarEnough == True:
                continue
            else:
                break
        if successes == len(words):
            candidates.pop(c)
            #ALLCHECKS.write(opentriple + "\t" + reasons + "\tLM"+ "\tLM_ALL\n")
            #ALLCHECKS.flush()
    return candidates

def improveexplanation(d, HYPERS, s):
    s_hypernyms = []
    if s in HYPERS:
        s_hypernyms = HYPERS[s]
    prov = ""
    common_hypernyms = {}
    common_hypernyms_count = {}
    common_hypernyms["others"] = []
    common_hypernyms_count["others"] = -1
    #print(str(details), flush=True)
    siblings = d
    #print(str(siblings), flush=True)
    for sib in siblings: #lion
        sib = sib.rstrip()
        if len(sib.rstrip()) == 0:
            continue
        if sib not in HYPERS: # [lion: mammal, large animal]
            others = common_hypernyms["others"]
            others.append(sib)
            common_hypernyms["others"] = others
            continue
        hypernyms = HYPERS[sib]
        if len(list(set(hypernyms) & set(s_hypernyms))) == 0:
            others = common_hypernyms["others"]
            others.append(sib)
            common_hypernyms["others"] = others
            continue
        for h in hypernyms:
            if h not in s_hypernyms:
                continue
            if h not in common_hypernyms:
                l = []
                l.append(sib)
                common_hypernyms[h] = l
                common_hypernyms_count[h] = 1
            else:
                l = common_hypernyms[h]
                l.append(sib)
                common_hypernyms[h] = l
                common_hypernyms_count[h] = common_hypernyms_count[h] + 1

    common_list = sortdic(common_hypernyms_count)
    seen = {}
    #print(str("1"), flush=True)
    for c in common_list:
        st = ""
        for sib in common_hypernyms[c[0]]:
            if sib not in seen:
                st = sib + ", " + st
                seen[sib] = 1
        if len(st) > 0:
            prov = prov + str(c[0]) + " :" + st +"# "
    #print(str("2"), flush=True)
    prov = prov + " others: " + str(common_hypernyms["others"]) + "# "
    return prov


def rescore(candidate_negations, hard, praphraser, KG_PO, details, total,sim_thresh, unique_details):
    soft = {}
    soft_details = {}
    candidates = []
    text_to_triple =  {}
    unique_peers = {}
    for c in hard:
        sentence = KG_PO[c].rstrip()
        sentence = sentence.replace('.','')
        sentence = sentence.replace('be ','')
        candidates.append(sentence)
        text_to_triple[sentence] = c
        soft[c] = 0
        soft_details[c] = ""
        unique_peers[c] = []
        if c in details:
            peers = details[c].rstrip().split(", ")
            #print("c filling "+str(c), flush=True)
            #print("filling peers "+str(peers), flush=True)
            unique_peers[c] = peers
    #print("unique peers "+str(unique_peers), flush = True)
    if len(candidates) == 0:
        return candidate_negations, soft, soft_details, unique_details

    paraphrases = util.paraphrase_mining(praphraser, candidates)

    for paraphrase in paraphrases:
        score, i, j = paraphrase
        c1 = candidates[i]
        c2 = candidates[j]
        if c1 == c2:
            continue
        if score < sim_thresh:
            #ALLCHECKS.write(str(c1) + "\t" + str(c2) + "\t"+ str(score) + "\tRESCORING\n")
            #ALLCHECKS.flush()
            continue
        else:
            #ALLCHECKS.write(str(c1) + "\t" + str(c2) + "\t"+ str(score) + "\tRESCORING\n")
            #ALLCHECKS.flush()
            key1 = text_to_triple[c1]
            key2 = text_to_triple[c2]
            peers1 = unique_peers[key1]
            #print("c1 "+str(c1), flush = True)
            #print("key1 "+str(key1), flush = True)
            #print("peers 1 "+str(peers1), flush = True)
            peers2 = unique_peers[key2]
            #print("c2 "+str(c2), flush = True)
            #print("key2 "+str(key2), flush = True)
            #print("peers 2 "+str(peers2), flush = True)
            newelements1 = list(set(peers2) - set(peers1))
            #print("new elements 1 "+str(newelements1), flush = True)
            newelements2 = list(set(peers1) - set(peers2))
            #print("new elements 2 "+str(newelements2), flush = True)
            if len(newelements1) > 0:
                soft[key1] = soft[key1] + len(newelements1)
                soft_details[key1] = soft_details[key1] + ', '.join(map(str, newelements1))  + " (" + text_to_triple[c2] + ") "
                peers1.extend(newelements1)
                #print("updated peers 1 B "+str(peers1))
                unique_peers[key1] = peers1
                unique_details[key1].extend(newelements1)
                #print(str(unique_details[key1]), flush=True)
                #print("updated peers 1 "+str(unique_peers[key1]))
            if len(newelements2) > 0:
                soft[key2] = soft[key2] + len(newelements2)
                soft_details[key2] = soft_details[key2] + ', '.join(map(str, newelements2)) + " (" + text_to_triple[c1] + ") "
                peers2.extend(newelements2)
                #print("updated peers 2 B "+str(peers2))
                unique_peers[key2] = peers2
                unique_details[key2].extend(newelements2)
                #print(str(unique_details[key2]), flush=True)
                #print("updated peers 2 "+str(unique_peers[key2]))
        unique_details[key1] = list(set(unique_details[key1]))
        unique_details[key2] = list(set(unique_details[key2]))
    for c in candidate_negations:
        #candidate_negations[c] = (hard[c])/total
        candidate_negations[c] = (hard[c] + soft[c])/total

    return candidate_negations, soft, soft_details, unique_details

#MAIN

#print('loading lms - bert and paraphrase mining')
location = '/GW/D5data-14/harnaout/peerbased_cskb/datasets'
outlocation = '/GW/D5data-14/harnaout/peerbased_cskb/relaxed/0'

unmasker = pipeline('fill-mask', model='/scratch/GW/pool0/harnaout/cskb/BERTLARGEUNCASED', top_k=50)
paraphraser = SentenceTransformer('paraphrase-MiniLM-L6-v2')
stopwords = [":","0o", "0s", "3a", "3b", "3d", "6b", "6o", "a", "A", "a1", "a2", "a3", "a4", "ab", "able", "about", "above", "abst", "ac", "accordance", "according", "accordingly", "across", "act", "actually", "ad", "added", "adj", "ae", "af", "affected", "affecting", "after", "afterwards", "ag", "again", "against", "ah", "ain", "aj", "al", "all", "allow", "allows", "almost", "alone", "along", "already", "also", "although", "always", "am", "among", "amongst", "amoungst", "amount", "an", "and", "announce", "another", "any", "anybody", "anyhow", "anymore", "anyone", "anyway", "anyways", "anywhere", "ao", "ap", "apart", "apparently", "appreciate", "approximately", "ar", "are", "aren", "arent", "arise", "around", "as", "aside", "ask", "asking", "at", "au", "auth", "av", "available", "aw", "away", "awfully", "ax", "ay", "az", "b", "B", "b1", "b2", "b3", "ba", "back", "bc", "bd", "be", "became", "been", "before", "beforehand", "beginnings", "behind", "below", "beside", "besides", "best", "between", "beyond", "bi", "bill", "biol", "bj", "bk", "bl", "bn", "both", "bottom", "bp", "br", "brief", "briefly", "bs", "bt", "bu", "but", "bx", "by", "c", "C", "c1", "c2", "c3", "ca", "call", "came", "can", "cannot", "cant", "cc", "cd", "ce", "certain", "certainly", "cf", "cg", "ch", "ci", "cit", "cj", "cl", "clearly", "cm", "cn", "co", "com", "come", "comes", "con", "concerning", "consequently", "consider", "considering", "could", "couldn", "couldnt", "course", "cp", "cq", "cr", "cry", "cs", "ct", "cu", "cv", "cx", "cy", "cz", "d", "D", "d2", "da", "date", "dc", "dd", "de", "definitely", "describe", "described", "despite", "detail", "df", "di", "did", "didn", "dj", "dk", "dl", "do", "does", "doesn", "doing", "don", "done", "down", "downwards", "dp", "dr", "ds", "dt", "du", "due", "during", "dx", "dy", "e", "E", "e2", "e3", "ea", "each", "ec", "ed", "edu", "ee", "ef", "eg", "ei", "eight", "eighty", "either", "ej", "el", "eleven", "else", "elsewhere", "em", "en", "end", "ending", "enough", "entirely", "eo", "ep", "eq", "er", "es", "especially", "est", "et", "et-al", "etc", "eu", "ev", "even", "ever", "every", "everybody", "everyone", "everything", "everywhere", "ex", "exactly", "example", "except", "ey", "f", "F", "f2", "fa", "far", "fc", "few", "ff", "fi", "fifteen", "fifth", "fify", "fill", "find", "fire", "five", "fix", "fj", "fl", "fn", "fo", "followed", "following", "follows", "for", "former", "formerly", "forth", "forty", "found", "four", "fr", "from", "front", "fs", "ft", "fu", "full", "further", "furthermore", "fy", "g", "G", "ga", "gave", "ge", "get", "gets", "getting", "gi", "give", "given", "gives", "giving", "gj", "gl", "go", "goes", "going", "gone", "got", "gotten", "gr", "greetings", "gs", "gy", "h", "H", "h2", "h3", "had", "hadn", "happens", "hardly", "has", "hasn", "hasnt", "have", "haven", "having", "he", "hed", "hello", "help", "hence", "here", "hereafter", "hereby", "herein", "heres", "hereupon", "hes", "hh", "hi", "hid", "hither", "hj", "ho", "hopefully", "how", "howbeit", "however", "hr", "hs", "http", "hu", "hundred", "hy", "i2", "i3", "i4", "i6", "i7", "i8", "ia", "ib", "ibid", "ic", "id", "ie", "if", "ig", "ignored", "ih", "ii", "ij", "il", "im", "immediately", "in", "inasmuch", "inc", "indeed", "index", "indicate", "indicated", "indicates", "information", "inner", "insofar", "instead", "interest", "into", "inward", "io", "ip", "iq", "ir", "is", "isn", "it", "itd", "its", "iv", "ix", "iy", "iz", "j", "J", "jj", "jr", "js", "jt", "ju", "just", "k", "K", "ke", "keep", "keeps", "kept", "kg", "kj", "km", "ko", "l", "L", "l2", "la", "largely", "last", "lately", "later", "latter", "latterly", "lb", "lc", "le", "least", "les", "less", "lest", "let", "lets", "lf", "like", "liked", "likely", "line", "little", "lj", "ll", "ln", "lo", "look", "looking", "looks", "los", "lr", "ls", "lt", "ltd", "m", "M", "m2", "ma", "made", "mainly", "make", "makes", "many", "may", "maybe", "me", "meantime", "meanwhile", "merely", "mg", "might", "mightn", "mill", "million", "mine", "miss", "ml", "mn", "mo", "more", "moreover", "most", "mostly", "move", "mr", "mrs", "ms", "mt", "mu", "much", "mug", "must", "mustn", "my", "n", "N", "n2", "na", "name", "namely", "nay", "nc", "nd", "ne", "near", "nearly", "necessarily", "neither", "nevertheless", "new", "next", "ng", "ni", "nine", "ninety", "nj", "nl", "nn", "no", "nobody", "non", "none", "nonetheless", "noone", "nor", "normally", "nos", "not", "noted", "novel", "now", "nowhere", "nr", "ns", "nt", "ny", "o", "O", "oa", "ob", "obtain", "obtained", "obviously", "oc", "od", "of", "off", "often", "og", "oh", "oi", "oj", "ok", "okay", "ol", "old", "om", "omitted", "on", "once", "one", "ones", "only", "onto", "oo", "op", "oq", "or", "ord", "os", "ot", "otherwise", "ou", "ought", "our", "out", "outside", "over", "overall", "ow", "owing", "own", "ox", "oz", "p", "P", "p1", "p2", "p3", "page", "pagecount", "pages", "par", "part", "particular", "particularly", "pas", "past", "pc", "pd", "pe", "per", "perhaps", "pf", "ph", "pi", "pj", "pk", "pl", "placed", "please", "plus", "pm", "pn", "po", "poorly", "pp", "pq", "pr", "predominantly", "presumably", "previously", "primarily", "probably", "promptly", "proud", "provides", "ps", "pt", "pu", "put", "py", "q", "Q", "qj", "qu", "que", "quickly", "quite", "qv", "r", "R", "r2", "ra", "ran", "rather", "rc", "rd", "re", "readily", "really", "reasonably", "recent", "recently", "ref", "refs", "regarding", "regardless", "regards", "related", "relatively", "research-articl", "respectively", "resulted", "resulting", "results", "rf", "rh", "ri", "right", "rj", "rl", "rm", "rn", "ro", "rq", "rr", "rs", "rt", "ru", "run", "rv", "ry", "s", "S", "s2", "sa", "said", "saw", "say", "saying", "says", "sc", "sd", "se", "sec", "second", "secondly", "section", "seem", "seemed", "seeming", "seems", "seen", "sent", "seven", "several", "sf", "shall", "shan", "shed", "shes", "show", "showed", "shown", "showns", "shows", "si", "side", "since", "sincere", "six", "sixty", "sj", "sl", "slightly", "sm", "sn", "so", "some", "somehow", "somethan", "sometime", "sometimes", "somewhat", "somewhere", "soon", "sorry", "sp", "specifically", "specified", "specify", "specifying", "sq", "sr", "ss", "st", "still", "stop", "strongly", "sub", "substantially", "successfully", "such", "sufficiently", "suggest", "sup", "sure", "sy", "sz", "t", "T", "t1", "t2", "t3", "take", "taken", "taking", "tb", "tc", "td", "te", "tell", "ten", "tends", "tf", "th", "than", "thank", "thanks", "thanx", "that", "thats", "the", "their", "theirs", "them", "themselves", "then", "thence", "there", "thereafter", "thereby", "thered", "therefore", "therein", "thereof", "therere", "theres", "thereto", "thereupon", "these", "they", "theyd", "theyre", "thickv", "thin", "think", "third", "this", "thorough", "thoroughly", "those", "thou", "though", "thoughh", "thousand", "three", "throug", "through", "throughout", "thru", "thus", "ti", "til", "tip", "tj", "tl", "tm", "tn", "to", "together", "too", "took", "top", "toward", "towards", "tp", "tq", "tr", "tried", "tries", "truly", "try", "trying", "ts", "tt", "tv", "twelve", "twenty", "twice", "two", "tx", "u", "U", "u201d", "ue", "ui", "uj", "uk", "um", "un", "under", "unfortunately", "unless", "unlike", "unlikely", "until", "unto", "uo", "up", "upon", "ups", "ur", "us", "used", "useful", "usefully", "usefulness", "using", "usually", "ut", "v", "V", "va", "various", "vd", "ve", "very", "via", "viz", "vj", "vo", "vol", "vols", "volumtype", "vq", "vs", "vt", "vu", "w", "W", "wa", "was", "wasn", "wasnt", "way", "we", "wed", "welcome", "well", "well-b", "went", "were", "weren", "werent", "what", "whatever", "whats", "when", "whence", "whenever", "where", "whereafter", "whereas", "whereby", "wherein", "wheres", "whereupon", "wherever", "whether", "which", "while", "whim", "whither", "who", "whod", "whoever", "whole", "whom", "whomever", "whos", "whose", "why", "wi", "widely", "with", "within", "without", "wo", "won", "wonder", "wont", "would", "wouldn", "wouldnt", "www", "x", "X", "x1", "x2", "x3", "xf", "xi", "xj", "xk", "xl", "xn", "xo", "xs", "xt", "xv", "xx", "y", "Y", "y2", "yes", "yet", "yj", "yl", "you", "youd", "your", "youre", "yours", "yr", "ys", "yt", "z", "Z", "zero", "zi", "zz"]
limit = 1000
total = 30
sim_thresh=0.75
th=0.05
#print('loading peers')
path = str(sys.argv[1])
fileid = str(sys.argv[2])
#print(str(path), flush=True)
E = readinput(path)
#print(str(E), flush = True)
PEER_GROUPS = getpeerembeddings(location+'/peers_embeddings_with_taxonomy_LARGE.tsv', total)
HYPERS = gethypernyms(location+'/WebisALOD_full.tsv')
#ALLCHECKS = open(outlocation+'/validation_'+fileid+'.tsv', 'a', encoding='UTF8')
#SPEED = open(outlocation+'/speed_'+fileid+'.tsv', 'a', encoding='UTF8')
originalkg = loadoriginalkg(location+'/ascentpp.csv')
#print('loading KG & thresholding', flush = True)
KG, KG_PO, triple_as_text, FREQ, TYPI = loadkg(location+'/ascentpp.csv',th,originalkg)
KG = thresholding_positives(KG,FREQ, 300, x)

#print('inferring')
entity_index = 0
for e in PEER_GROUPS:
    start = timeit.default_timer()
    if e in E:
        group_members = PEER_GROUPS[e]
        total = len(group_members)
        #print("group size "+str(len(group_members)), flush = True)
        if e not in KG:
            stop = timeit.default_timer()
            #SPEED.write(str(e) + "\t" + str(stop - start) + "\n")
            #SPEED.flush()
            continue


        entity_index = entity_index + 1
        #if entity_index == 2:
        #    break
        positive_info = KG[e]

        #print('collect candidates',flush=True)
        candidate_negations, details, hard, validation_details, unique_details = collectcandidates(e, positive_info, KG, group_members, total)
        num_of_candidates_initial = len(candidate_negations)

        #print('paraphrase mining', flush = True)
        candidate_negations, validation_details = paraphrase_validation(candidate_negations, KG_PO, positive_info, paraphraser, validation_details, sim_thresh)
        #num_of_candidates_postparaphrasing = len(candidate_negations)
        #print("here0", flush=True)
        #print('lm probing', flush = True)
        #candidate_negations = lm_all(e, candidate_negations, KG_PO, ALLCHECKS, unmasker, paraphraser, stopwords)
        candidate_negations = lm_subject(e, candidate_negations, KG_PO, unmasker, paraphraser, stopwords,sim_thresh)
        #num_of_candidates_postlm = len(candidate_negations)
        #print("here1", flush=True)
        #print('soft scoring', flush = True)
        candidate_negations, soft, soft_details, unique_details= rescore(candidate_negations, hard, paraphraser, KG_PO, details, total,0.8, unique_details)
        #print("here2", flush=True)
        #print('sorting candidates')
        negations_list = sortdic(candidate_negations)
        #random.shuffle(negations_list)
        rank = 0
        #print('printing')
        for candidate in negations_list:
            #if rank == limit:
            #    break
            #print(str("provbefore"), flush=True)
            provenance = improveexplanation(unique_details[candidate[0]], HYPERS, e)
            #print("here3", flush=True)
            #print(str("provafter"), flush=True)
            #print(e + "\t¬(" + candidate[0] + ")\t" + str(candidate[1]) + "\t" + str(hard[candidate[0]]) + "\t" + provenance +"\t" +str(KG_PO[candidate[0]]), flush = True)
            print(e + "\t¬(" + candidate[0] + ")\t" + str(candidate[1]) + "\t" + str(hard[candidate[0]]) + "\t" + details[candidate[0]] + "\t" + str(soft[candidate[0]]) + "\t" + soft_details[candidate[0]] + "\t" + provenance +"\t" +str(KG_PO[candidate[0]]), flush = True)
            #print(e + "\t¬(" + candidate[0] + ")\t" + str(candidate[1]) + "\t" + str(rank+1) +"\t"+ str(hard[candidate[0]]) + "\t" + details[candidate[0]] + "\t" + str(soft[candidate[0]]) + "\t" + soft_details[candidate[0]] + "\t"+ str(positive_info) +"\t" + str(len(positive_info)) + "\t" + str(validation_details[candidate[0]]) + "\t" +str(num_of_candidates_initial) + "\t"+ str(num_of_candidates_postparaphrasing) + "\t" + str(num_of_candidates_postlm) +"\t"+str(KG_PO[candidate[0]]), flush = True)
            rank = rank + 1
        stop = timeit.default_timer()
        #SPEED.write(str(e) + "\t" + str(stop - start) + "\n")
        #SPEED.flush()