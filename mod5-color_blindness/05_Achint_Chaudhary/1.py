import pandas as pd
import numpy as np
import sympy
from copy import deepcopy

fp_bwt = open("../data/chrX_last_col.txt","r")
fp_res = open("../data/chrX.fa","r")

bwt_lines,ref_lines = fp_bwt.readlines(), fp_res.readlines()

bwt = ''.join(bwt_lines).replace('\n','')
ref = ''.join(ref_lines[1:]).replace('\n','')

fp_bwt.close()
fp_res.close()

chars = ['A','C','G','T']
count = dict.fromkeys(range(1+len(chars)) , 0)

for t in bwt:
    if t in ('A',):
        count[1]+=1
    if t in ('C','A'):
        count[2]+=1
    if t in ('G','C','A'):
        count[3]+=1
    if t in ('T','G','C','A'):
        count[4]+=1

for k in count:
    print(k,count[k])


occ = dict.fromkeys(chars,[])
for key in occ.keys():
    occ[key] = deepcopy(occ[key])

cum_count = dict((x,0) for x in chars)

for bwt_ix,bwt_ch in enumerate(bwt):
    for chars_ch in chars:
        if bwt_ch == chars_ch:
            cum_count[chars_ch] += 1
        occ[chars_ch].append( cum_count[chars_ch] )

#     if bwt_ix%10==0:
#         for chars_ch in chars:
#             occ[chars_ch].append( cum_count[chars_ch] )

for i in chars:
    print(list(occ[i][:150:10]))


def Occurrence(ch,index):
    return occ[ch][index] if index>=0 else 0

def BW_search(read):
    i=len(read)-1
    c1 = read[i]
    c2 = chars.index(c1) if (c1 in chars) else len(chars)
    sp, ep = count[c2], count[c2+1]-1
    while i:
        if sp<=ep:
            c1 = read[i-1]
            c2 = chars.index(c1) if (c1 in chars) else len(chars)
            sp, ep = count[c2]+Occurrence(c1,sp-1), count[c2]+Occurrence(c1,ep)-1
            i-=1
    return (-np.inf,-np.inf) if(ep<sp) else (sp,ep)


chrX_index_map = pd.read_csv("../data/chrX_map.txt",squeeze=True,header=None)

r_ex = dict.fromkeys([1,2,3,4,5,6], 0)
g_ex = dict.fromkeys([1,2,3,4,5,6], 0)


range_r_ex={
    1:[149249757,149249868],
    2:[149256127,149256423],
    3:[149258412,149258580],
    4:[149260048,149260213],
    5:[149261768,149262007],
    6:[149264290,149264400]
        }

range_g_ex={
    1:[149288166,149288277],
    2:[149293258,149293554],
    3:[149295542,149295710],
    4:[149297178,149297343],
    5:[149298898,149299137],
    6:[149301420,149301530]
        }


def add_count(l):
    if len(l)==2:
        z=0.5
    else:
        z=1
    for x in l:
        for key,val in range_r_ex.items():
            if x>=val[0] and x<=val[1]:
                #print key
                r_ex[key]+=z
        for key,val in range_g_ex.items():
            if x>=val[0] and x<=val[1]:
                #print key
                g_ex[key]+=z



with open("../data/reads","r") as fp_reads:
    all_lines = fp_reads.readlines()
    for line in all_lines[2900000:]:
        line=line[:-1]
        line=line.replace('N','A')
        n=len(line)
        r1=line[:n//3]
        r2=line[n//3:2*n//3]
        r3=line[2*n//3:]
        sp1,ep1=BW_search(r1)
        sp2,ep2=BW_search(r2)
        sp3,ep3=BW_search(r3)
        len1=len(r1)
        len2=len(r2)
        len3=len(r3)
        dup=[]

        if (sp1<0 or ep1<0) and (sp2<0 or ep2<0) and (sp3<0 or ep3<0):
            continue

        if sp1>=0 and ep1>=0:
            l1=[]
            for j in range(sp1,ep1+1):
                d=chrX_index_map[j]
                if d>149249750 and d<149301535:
                    l1.append(d)

            for x in l1:
                mismatch=0
                u=n//3
                for p in range(x+len1,x+len1+len2+len3):
                    if line[u]!=ref[p]:
                        mismatch+=1
                    u+=1
                if mismatch<=2:
                    dup.append(x)

        if sp2>=0 and ep2>=0:
            l1=[]
            for j in range(sp2,ep2+1):
                d=chrX_index_map[j]
                if d>149248700 and d<149302595:
                    l1.append(d)
            for x in l1:
                mismatch=0
                v=2*n//3
                u=0
                for p in range(x-n//3,x):
                    if line[u]!=ref[p]:
                        mismatch+=1
                    u+=1
                for p in range(x+len2,x+len2+len3):
                    if line[v]!=ref[p]:
                        mismatch+=1
                    v+=1
                if mismatch<=2:
                    x1=x-n//3
                    if x1 not in dup:
                        dup.append(x1)

        if sp3>=0 and ep3>=0:
            l1=[]
            for j in range(sp3,ep3+1):
                d=chrX_index_map[j]
                if d>149248700 and d<149302595:
                    l1.append(d)
            for x in l1:
                mismatch=0
                u=0
                for p in range(x-2*n//3,x):
                    if line[u]!=ref[p]:
                        mismatch+=1
                    u+=1
                if mismatch<=2:
                    x1=x-2*n//3
                    if x1 not in dup:
                        dup.append(x1)
    
        dup.sort()
        if dup:
            add_count(dup)



prob_of_models = {
    1:[ ( 1.0/3.0  ,   2.0/3.0  )  ,  ( 1.0/3.0  ,2.0/3.0    )  ,  ( 1.0/3.0  ,   2.0/3.0  )  ,  (1.0/3.0, 2.0/3.0) ],
    2:[ ( 0.5/1.0  ,   0.5/1.0  )  ,  ( 0.5/1.0  ,0.5/1.0    )  ,  ( 0.0/1.0  ,   1.0/1.0  )  ,  (0.0/1.0, 1.0/1.0) ],
    3:[ (33.0/133.0, 100.0/133.0)  ,  (33.0/133.0,100.0/133.0)  ,  ( 0.5/1.0  ,   0.5/1.0  )  ,  (0.5/1.0, 0.5/1.0) ],
    4:[ (33.0/133.0, 100.0/133.0)  ,  (33.0/133.0,100.0/133.0)  ,  (33.0/133.0, 100.0/133.0)  ,  (0.5/1.0, 0.5/1.0) ],
     }




prob_list = np.zeros(4)
for exon in range(2,6):
    r=r_ex[exon]
    g=g_ex[exon]
    total=r+g
    for model,prob in prob_of_models.items():
        succ_prob = prob[exon-2][0]**r
        fail_prob = prob[exon-2][1]**g
        prob_list[model-1] += np.log( float(sympy.binomial(total,r)*succ_prob*fail_prob) )


maximum_pobability=max(prob_list)
    
for i,prob in enumerate(prob_list):
    if prob==maximum_pobability:
        model=i+1
    print('Probability of model '+str(i+1)+' is',prob)



print('\nThe model having highest probability is', model)