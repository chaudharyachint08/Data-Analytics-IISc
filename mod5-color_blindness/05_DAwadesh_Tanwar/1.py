# Below codes takes 1873.628seconds (31.2 minutes) for execution
import warnings, os
warnings.filterwarnings('ignore')
import numpy as np,pandas as pd
from sympy import binomial as dist

mlstn_delta = 10
dct = {'A':0,'C':1,'G':2,'T':3,'$':4}
xn_red, xn_green = dict.fromkeys(range(1,6+1), 0), dict.fromkeys(range(1,6+1), 0)

ex_red_intrvl={
    1:[149249757,149249868], 2:[149256127,149256423],
    3:[149258412,149258580], 4:[149260048,149260213],
    5:[149261768,149262007], 6:[149264290,149264400]
        }

ex_green_intrvl={
    1:[149288166,149288277], 2:[149293258,149293554],
    3:[149295542,149295710], 4:[149297178,149297343],
    5:[149298898,149299137], 6:[149301420,149301530]
        }

model_prob = {
    0:[ (1/3,2/3) , (1/3,2/3) , (1/3,2/3) , (1/3,2/3) ],
    1:[ (1/2,1/2) , (1/2,1/2) , (0/1,1/1) , (0/1,1/1) ],
    2:[ (1/4,3/4) , (1/4,3/4) , (1/2,1/2) , (1/2,1/2) ],
    3:[ (1/4,3/4) , (1/4,3/4) , (1/4,3/4) , (1/2,1/2) ],
     }

sym_lm1, sym_lm2, sym_lm3, sym_lm4 = 149249750, 149301535, 149248700, 149302595


cX_ix_mp = np.array( pd.read_csv(os.path.join('..','data','chrX_map.txt'),squeeze=True,header=None,dtype=np.int32) )

with open(os.path.join('..','data','chrX_last_col.txt')) as bwt_file_handle, open(os.path.join('..','data','chrX.fa')) as fp_ref:
    bw_str, rf_str = ''.join(i.strip() for i in bwt_file_handle.readlines()), ''.join(i.strip() for i in fp_ref.readlines()[1:])
# VERIFYING PROCESS
# print(len(bw_str),len(rf_str))

count_A, count_C, count_G, count_T = bw_str.count('A'), bw_str.count('C'), bw_str.count('G'), bw_str.count('T')
agg_val = {}
agg_val[0] = 0
agg_val[1] = 0 + count_A
agg_val[2] = 0 + count_A + count_C
agg_val[3] = 0 + count_A + count_C + count_G
agg_val[4] = 0 + count_A + count_C + count_G + count_T
# VERIFYING PROCESS
# for k in agg_val:
#     print(k,agg_val[k])

mlstn_cmlt = { x:np.cumsum(1*(np.array(list(bw_str))==x))[::mlstn_delta] for x in ('A','C','G','T') }
# VERIFYING PROCESS
# for i in ['A','C','G','T']:
#     print(list(mlstn_cmlt[i][:20]))


def occ(c,ix):
    return (mlstn_cmlt[c][ix//mlstn_delta]+(bw_str[ix-ix%mlstn_delta+1:ix+1]).count(c)) if ix>=0 else 0

def bws(read):
    i=len(read)-1
    c1 = read[i]
    c  = dct[c1]
    sp=agg_val[c]
    ep=agg_val[c+1]-1    
    while i and sp<=ep:
        c1=read[i-1]
        c =dct[c1]
        sp=agg_val[c]+occ(c1,sp-1)
        ep=agg_val[c]+occ(c1,ep)-1
        i-=1
    return ((sp,ep) if (ep>=sp) else (-1,-1))


def make_entries(l):
    z = 0.5 if len(l)==2 else 1
    for key, val in ex_red_intrvl.items():
        xn_red[key]   += z*(((val[0]<=np.array(l))&(np.array(l)<=val[1])).sum())
    for key, val in ex_green_intrvl.items():
        xn_green[key] += z*(((val[0]<=np.array(l))&(np.array(l)<=val[1])).sum())

rf_str = np.array(list(rf_str))
with open(os.path.join('..','data','reads')) as fp_reads:    
    all_lines = fp_reads.readlines()#[2900000:3066721]
    for read in all_lines:
        read, repeat_set = np.array(list(read.strip().replace('N','A'))), set()
        len_read = len(read)
        read1, read2, read3 = read[:len_read//3] , read[len_read//3:2*len_read//3] , read[2*len_read//3:]
        len1,len2,len3 = len(read1), len(read2), len(read3)
        (sp1,ep1), (sp2,ep2), (sp3,ep3) = bws(read1), bws(read2), bws(read3)

        if (np.array([sp1,ep1,sp2,ep2,sp3,ep3])<0).all():
            continue

        if (np.array([sp1,ep1])>=0).all():
            map_slc = cX_ix_mp[sp1:ep1+1]
            val_ix  = ((map_slc>sym_lm1)&(map_slc<sym_lm2))
            mismatch_list = map_slc[val_ix]
            lwr_ix = len_read//3
            for x in mismatch_list:
                lm1, lm2 = x+len1, x+len1+len2+len3
                count_mismatch  = (rf_str[lm1:lm2]!=read[lwr_ix:lwr_ix+(lm2-lm1)]).sum()
                if count_mismatch<=2:
                    repeat_set.add(x)

        if (np.array([sp2,ep2])>=0).all():
            map_slc = cX_ix_mp[sp2:ep2+1]
            val_ix  = ((map_slc>sym_lm3)&(map_slc<sym_lm4))
            mismatch_list = map_slc[val_ix]
            lwr_ix, upr_ix = 0, 2*len_read//3
            for x in mismatch_list:
                lm1, lm2, lm3, lm4 =  x-len_read//3, x, x+len2, x+len2+len3
                count_mismatch = (rf_str[lm1:lm2]!=read[lwr_ix:lwr_ix+(lm2-lm1)]).sum() + (rf_str[lm3:lm4]!=read[upr_ix:upr_ix+(lm4-lm3)]).sum()
                if count_mismatch<=2:
                    repeat_set.add( x-len_read//3 )

        if (np.array([sp3,ep3])>=0).all():
            map_slc = cX_ix_mp[sp3:ep3+1]
            val_ix  = ((map_slc>sym_lm3)&(map_slc<sym_lm4))
            mismatch_list = map_slc[val_ix]
            lwr_ix = 0
            for x in mismatch_list:
                lm1, lm2 = x-2*len_read//3, x
                count_mismatch  = (rf_str[lm1:lm2]!=read[lwr_ix:lwr_ix+(lm2-lm1)]).sum()
                if count_mismatch<=2:
                    repeat_set.add(x-2*len_read//3)

        if repeat_set:
            make_entries( sorted(repeat_set) )

    for key in xn_red:
        xn_red[key]   /= len(xn_red)
        xn_green[key] /= len(xn_green)


# VERIFYING PROCESS
# print(xn_red,xn_green,sep='\n')

prob_list = np.zeros(4)
for mdl,prob in model_prob.items():
    prob_list[mdl] = sum( np.log(float(dist(xn_red[exon+2]+xn_green[exon+2],xn_red[exon+2])*(prob[exon][0]**xn_red[exon+2])*(prob[exon][1]**xn_green[exon+2]))) for exon in range(4))

print('Models with their respective log-probabilities')
print(*enumerate(prob_list,start=1),sep='\n',end='\n\n\n')
print('The model having highest probability is')
print( max(enumerate(prob_list,start=1),key=lambda x:x[1])[0] )